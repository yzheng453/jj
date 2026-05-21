// Copyright 2024 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;

use thiserror::Error;

use crate::eval_types::{EvalType, FunctionSignature};
use crate::eval::ExpressionNode;

#[derive(Debug, Error)]
pub enum TypeError {
    #[error("Type mismatch: expected {expected:?}, got {actual:?}")]
    TypeMismatch { expected: EvalType, actual: EvalType },
    #[error("Undefined function or method: {0}")]
    UndefinedFunction(String),
    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),
    #[error("Generic error: {0}")]
    GenericError(String),
}

#[derive(Clone, Debug)]
pub struct TypedExpressionNode<'i> {
    pub kind: TypedExpressionKind<'i>,
    pub ty: EvalType,
    pub span: pest::Span<'i>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarKind {
    Capture,
    Arg,
    Local,
}

#[derive(Clone, Debug)]
pub enum TypedExpressionKind<'i> {
    LocalVariable(VarKind, usize),
    Boolean(bool),
    Integer(i64),
    String(std::borrow::Cow<'i, str>),
    FunctionCall(Box<TypedFunctionCallNode<'i>>),
    Lambda(Box<TypedLambdaNode<'i>>),
    Unary(crate::eval::UnaryOp, Box<TypedExpressionNode<'i>>),
    Binary(crate::eval::BinaryOp, Box<TypedExpressionNode<'i>>, Box<TypedExpressionNode<'i>>),
    MethodCall(Box<TypedFunctionCallNode<'i>>),
    Binding(usize, Box<TypedExpressionNode<'i>>), // binding now only applies to Locals
    Sequence(Vec<TypedExpressionNode<'i>>),
}

#[derive(Clone, Debug)]
pub struct TypedFunctionCallNode<'i> {
    pub function_name: String,
    pub args: Vec<TypedExpressionNode<'i>>,
}

#[derive(Clone, Debug)]
pub struct Capture {
    pub ty: EvalType,
    pub outer_kind: VarKind,
    pub outer_index: usize,
}

#[derive(Clone, Debug)]
pub struct TypedLambdaNode<'i> {
    pub captures: Vec<Capture>,
    pub num_args: usize,
    pub body: TypedExpressionNode<'i>,
}

pub struct LambdaFrame<'i> {
    pub scopes: Vec<HashMap<&'i str, (EvalType, VarKind, usize)>>,
    pub captures: Vec<Capture>,
    pub num_args: usize,
    pub next_local: usize,
}

impl<'i> LambdaFrame<'i> {
    pub fn new(num_args: usize) -> Self {
        Self {
            scopes: vec![HashMap::new()],
            captures: Vec::new(),
            num_args,
            next_local: 0,
        }
    }
}

use std::sync::OnceLock;

pub struct TypeContext<'i> {
    pub frames: Vec<LambdaFrame<'i>>,
}

pub fn builtin_functions() -> &'static HashMap<String, FunctionSignature> {
    static BUILTINS: OnceLock<HashMap<String, FunctionSignature>> = OnceLock::new();
    BUILTINS.get_or_init(|| {
        let mut functions = HashMap::new();
        functions.insert(
            "abandon".to_string(),
            FunctionSignature {
                params: vec![EvalType::Commit],
                ret: EvalType::Unit,
                mutates: true,
            },
        );
        functions.insert(
            "map_int".to_string(),
            FunctionSignature {
                params: vec![EvalType::Integer, EvalType::Lambda(vec![EvalType::Integer], Box::new(EvalType::Integer))],
                ret: EvalType::Integer,
                mutates: false,
            },
        );
        functions.insert(
            "map_bool".to_string(),
            FunctionSignature {
                params: vec![EvalType::Integer, EvalType::Lambda(vec![EvalType::Integer], Box::new(EvalType::Boolean))],
                ret: EvalType::Boolean,
                mutates: false,
            },
        );
        functions
    })
}

impl<'i> TypeContext<'i> {
    pub fn default_builtins() -> Self {
        Self {
            frames: vec![LambdaFrame::new(0)],
        }
    }

    pub fn resolve_variable(&mut self, name: &'i str) -> Result<(EvalType, VarKind, usize), TypeError> {
        for frame_idx in (0..self.frames.len()).rev() {
            for scope in self.frames[frame_idx].scopes.iter().rev() {
                if let Some(&(ref ty, kind, index)) = scope.get(name) {
                    let current_ty = ty.clone();
                    let mut current_kind = kind;
                    let mut current_index = index;
                    
                    // Thread it through intermediate frames
                    for intermediate_idx in (frame_idx + 1)..self.frames.len() {
                        let frame = &mut self.frames[intermediate_idx];
                        let pos = frame.captures.iter().position(|c| c.outer_kind == current_kind && c.outer_index == current_index);
                        current_index = if let Some(p) = pos {
                            p
                        } else {
                            let new_idx = frame.captures.len();
                            frame.captures.push(Capture {
                                ty: current_ty.clone(),
                                outer_kind: current_kind,
                                outer_index: current_index,
                            });
                            new_idx
                        };
                        current_kind = VarKind::Capture;
                    }
                    
                    return Ok((current_ty, current_kind, current_index));
                }
            }
        }
        Err(TypeError::UndefinedVariable(name.to_string()))
    }
}

pub fn unify(
    expected: &EvalType,
    actual: &EvalType,
    subs: &mut HashMap<usize, EvalType>,
) -> Result<(), TypeError> {
    match (expected, actual) {
        (EvalType::TypeVar(id), _) => {
            if let Some(existing) = subs.get(id).cloned() {
                unify(&existing, actual, subs)
            } else {
                subs.insert(*id, actual.clone());
                Ok(())
            }
        }
        (_, EvalType::TypeVar(id)) => {
            if let Some(existing) = subs.get(id).cloned() {
                unify(expected, &existing, subs)
            } else {
                subs.insert(*id, expected.clone());
                Ok(())
            }
        }
        (EvalType::Unit, EvalType::Unit) => Ok(()),
        (EvalType::Boolean, EvalType::Boolean) => Ok(()),
        (EvalType::Integer, EvalType::Integer) => Ok(()),
        (EvalType::String, EvalType::String) => Ok(()),
        (EvalType::Commit, EvalType::Commit) => Ok(()),
        (EvalType::ChangeId, EvalType::ChangeId) => Ok(()),
        (EvalType::List(e), EvalType::List(a)) => unify(e, a, subs),
        (EvalType::Map(ek, ev), EvalType::Map(ak, av)) => {
            unify(ek, ak, subs)?;
            unify(ev, av, subs)
        }
        (EvalType::Lambda(e_params, e_ret), EvalType::Lambda(a_params, a_ret)) => {
            if e_params.len() != a_params.len() {
                return Err(TypeError::TypeMismatch { expected: expected.clone(), actual: actual.clone() });
            }
            for (ep, ap) in e_params.iter().zip(a_params.iter()) {
                unify(ep, ap, subs)?;
            }
            unify(e_ret, a_ret, subs)
        }
        _ => Err(TypeError::TypeMismatch {
            expected: expected.clone(),
            actual: actual.clone(),
        }),
    }
}

fn is_hashable(ty: &EvalType) -> bool {
    match ty {
        EvalType::Lambda(_, _) | EvalType::Map(_, _) | EvalType::List(_) => false,
        _ => true,
    }
}

pub fn apply_substitutions(ty: &EvalType, subs: &HashMap<usize, EvalType>) -> Result<EvalType, TypeError> {
    match ty {
        EvalType::TypeVar(id) => {
            if let Some(sub) = subs.get(id) {
                apply_substitutions(sub, subs)
            } else {
                Ok(ty.clone())
            }
        }
        EvalType::List(inner) => Ok(EvalType::List(Box::new(apply_substitutions(inner, subs)?))),
        EvalType::Map(k, v) => {
            let key_ty = apply_substitutions(k, subs)?;
            if !is_hashable(&key_ty) {
                return Err(TypeError::GenericError(format!("Type {:?} cannot be used as a Map key", key_ty)));
            }
            let val_ty = apply_substitutions(v, subs)?;
            Ok(EvalType::Map(Box::new(key_ty), Box::new(val_ty)))
        }
        EvalType::Lambda(params, ret) => {
            let resolved_params: Result<Vec<_>, _> = params.iter().map(|p| apply_substitutions(p, subs)).collect();
            Ok(EvalType::Lambda(resolved_params?, Box::new(apply_substitutions(ret, subs)?)))
        }
        _ => Ok(ty.clone()),
    }
}

pub fn type_check_with_expected<'i>(
    node: &ExpressionNode<'i>,
    expected: &EvalType,
    ctx: &mut TypeContext<'i>,
) -> Result<TypedExpressionNode<'i>, TypeError> {
    if let crate::eval::ExpressionKind::Lambda(lambda) = &node.kind {
        if let EvalType::Lambda(param_tys, _) = expected {
            if lambda.params.len() != param_tys.len() {
                return Err(TypeError::GenericError("Lambda arity mismatch".to_string()));
            }
            
            ctx.frames.push(LambdaFrame::new(lambda.params.len()));
            
            for (i, (name, _explicit_ty)) in lambda.params.iter().enumerate() {
                ctx.frames.last_mut().unwrap().scopes.last_mut().unwrap().insert(*name, (param_tys[i].clone(), VarKind::Arg, i));
            }
            
            let typed_body = type_check(&lambda.body, ctx)?;
            
            let frame = ctx.frames.pop().unwrap();
            
            let body_ty = typed_body.ty.clone();
            let typed_lambda = TypedLambdaNode {
                captures: frame.captures,
                num_args: lambda.params.len(),
                body: typed_body,
            };
            
            return Ok(TypedExpressionNode {
                kind: TypedExpressionKind::Lambda(Box::new(typed_lambda)),
                ty: EvalType::Lambda(param_tys.clone(), Box::new(body_ty)),
                span: node.span.clone(),
            });
        }
    }
    // Fallback to normal type_check
    type_check(node, ctx)
}

pub fn type_check<'i>(
    node: &ExpressionNode<'i>,
    ctx: &mut TypeContext<'i>,
) -> Result<TypedExpressionNode<'i>, TypeError> {
    let (kind, ty) = match &node.kind {
        crate::eval::ExpressionKind::Integer(i) => (TypedExpressionKind::Integer(*i), EvalType::Integer),
        crate::eval::ExpressionKind::Boolean(b) => (TypedExpressionKind::Boolean(*b), EvalType::Boolean),
        crate::eval::ExpressionKind::String(s) => (TypedExpressionKind::String(std::borrow::Cow::Owned(s.clone())), EvalType::String),
        crate::eval::ExpressionKind::Identifier(name) => {
            let (ty, kind, index) = ctx.resolve_variable(name)?;
            (TypedExpressionKind::LocalVariable(kind, index), ty)
        }
        crate::eval::ExpressionKind::FunctionCall(call) => {
            let sig = builtin_functions().get(call.name).ok_or_else(|| {
                TypeError::UndefinedFunction(call.name.to_owned())
            })?;
            
            if call.args.len() != sig.params.len() {
                return Err(TypeError::UndefinedFunction(format!("Arity mismatch for {}", call.name)));
            }
            
            let mut subs = HashMap::new();
            let mut typed_args = Vec::new();
            
            for (arg, param_ty) in call.args.iter().zip(sig.params.iter()) {
                let expected_ty = apply_substitutions(param_ty, &subs)?;
                let typed_arg = type_check_with_expected(arg, &expected_ty, ctx)?;
                unify(&expected_ty, &typed_arg.ty, &mut subs)?;
                typed_args.push(typed_arg);
            }
            
            let ret_ty = apply_substitutions(&sig.ret, &subs)?;
            let typed_call = TypedFunctionCallNode {
                function_name: call.name.to_owned(),
                args: typed_args,
            };
            (TypedExpressionKind::FunctionCall(Box::new(typed_call)), ret_ty)
        }
        crate::eval::ExpressionKind::Unary(op, arg) => {
            let typed_arg = type_check(arg, ctx)?;
            let ty = match op {
                crate::eval::UnaryOp::LogicalNot => {
                    if typed_arg.ty != EvalType::Boolean {
                        return Err(TypeError::TypeMismatch { expected: EvalType::Boolean, actual: typed_arg.ty });
                    }
                    EvalType::Boolean
                }
                crate::eval::UnaryOp::Negate => {
                    if typed_arg.ty != EvalType::Integer {
                        return Err(TypeError::TypeMismatch { expected: EvalType::Integer, actual: typed_arg.ty });
                    }
                    EvalType::Integer
                }
            };
            (TypedExpressionKind::Unary(*op, Box::new(typed_arg)), ty)
        }
        crate::eval::ExpressionKind::Binary(op, lhs, rhs) => {
            let typed_lhs = type_check(lhs, ctx)?;
            let typed_rhs = type_check(rhs, ctx)?;
            if typed_lhs.ty != typed_rhs.ty {
                return Err(TypeError::TypeMismatch { expected: typed_lhs.ty.clone(), actual: typed_rhs.ty.clone() });
            }
            let ty = match op {
                crate::eval::BinaryOp::LogicalAnd | crate::eval::BinaryOp::LogicalOr => {
                    if typed_lhs.ty != EvalType::Boolean {
                        return Err(TypeError::TypeMismatch { expected: EvalType::Boolean, actual: typed_lhs.ty });
                    }
                    EvalType::Boolean
                }
                crate::eval::BinaryOp::Eq | crate::eval::BinaryOp::Ne => EvalType::Boolean,
                crate::eval::BinaryOp::Gt | crate::eval::BinaryOp::Ge | crate::eval::BinaryOp::Lt | crate::eval::BinaryOp::Le => {
                    if typed_lhs.ty != EvalType::Integer {
                        return Err(TypeError::TypeMismatch { expected: EvalType::Integer, actual: typed_lhs.ty });
                    }
                    EvalType::Boolean
                }
                crate::eval::BinaryOp::Add | crate::eval::BinaryOp::Sub | crate::eval::BinaryOp::Mul | crate::eval::BinaryOp::Div | crate::eval::BinaryOp::Rem => {
                    if typed_lhs.ty != EvalType::Integer {
                        return Err(TypeError::TypeMismatch { expected: EvalType::Integer, actual: typed_lhs.ty });
                    }
                    EvalType::Integer
                }
            };
            (TypedExpressionKind::Binary(*op, Box::new(typed_lhs), Box::new(typed_rhs)), ty)
        }
        crate::eval::ExpressionKind::MethodCall(method) => {
            let sig = builtin_functions().get(method.function.name).ok_or_else(|| {
                TypeError::UndefinedFunction(method.function.name.to_owned())
            })?;
            
            if method.function.args.len() + 1 != sig.params.len() {
                return Err(TypeError::UndefinedFunction(format!("Arity mismatch for {}", method.function.name)));
            }
            
            let mut subs = HashMap::new();
            let mut typed_args = Vec::new();
            
            let typed_receiver = type_check(&method.object, ctx)?;
            unify(&sig.params[0], &typed_receiver.ty, &mut subs)?;
            typed_args.push(typed_receiver);
            
            for (arg, param_ty) in method.function.args.iter().zip(sig.params.iter().skip(1)) {
                let expected_ty = apply_substitutions(param_ty, &subs)?;
                let typed_arg = type_check_with_expected(arg, &expected_ty, ctx)?;
                unify(&expected_ty, &typed_arg.ty, &mut subs)?;
                typed_args.push(typed_arg);
            }
            
            let ret_ty = apply_substitutions(&sig.ret, &subs)?;
            let typed_call = TypedFunctionCallNode {
                function_name: method.function.name.to_owned(),
                args: typed_args,
            };
            (TypedExpressionKind::FunctionCall(Box::new(typed_call)), ret_ty)
        }
        crate::eval::ExpressionKind::Binding(binding) => {
            let typed_val = type_check(&binding.value, ctx)?;
            let frame = ctx.frames.last_mut().unwrap();
            let slot = frame.next_local;
            frame.next_local += 1;
            frame.scopes.last_mut().unwrap().insert(binding.name, (typed_val.ty.clone(), VarKind::Local, slot));
            
            (TypedExpressionKind::Binding(slot, Box::new(typed_val)), EvalType::Unit)
        }
        crate::eval::ExpressionKind::Sequence(nodes) => {
            ctx.frames.last_mut().unwrap().scopes.push(HashMap::new());
            let mut sequence_nodes = Vec::new();
            for node in nodes {
                let typed_node = type_check(node, ctx)?;
                sequence_nodes.push(typed_node);
            }
            ctx.frames.last_mut().unwrap().scopes.pop();
            let last_ty = sequence_nodes.last().map(|n| n.ty.clone()).unwrap_or(EvalType::Unit);
            (TypedExpressionKind::Sequence(sequence_nodes), last_ty)
        }
        crate::eval::ExpressionKind::Lambda(_) => {
            return Err(TypeError::GenericError("Cannot infer type of lambda without context".to_string()));
        }
        _ => return Err(TypeError::GenericError("Unsupported node".to_owned()))
    };

    Ok(TypedExpressionNode {
        kind,
        ty,
        span: node.span.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typecheck_literals() {
        let mut ctx = TypeContext::default_builtins();
        let text = "42";
        let raw_ast = crate::eval::parse_eval_expressions(text).unwrap();
        let typed_ast = type_check(&raw_ast, &mut ctx).unwrap();
        assert_eq!(typed_ast.ty, EvalType::Integer);
        assert!(matches!(typed_ast.kind, TypedExpressionKind::Integer(42)));
    }

    #[test]
    fn test_typecheck_bindings() {
        let mut ctx = TypeContext::default_builtins();
        let text = "let x = 1; x";
        let raw_ast = crate::eval::parse_eval_expressions(text).unwrap();
        let typed_ast = type_check(&raw_ast, &mut ctx).unwrap();
        
        if let TypedExpressionKind::Sequence(nodes) = typed_ast.kind {
            assert_eq!(nodes.len(), 2);
            if let TypedExpressionKind::Binding(slot, val) = &nodes[0].kind {
                assert_eq!(*slot, 0);
                assert!(matches!(val.kind, TypedExpressionKind::Integer(1)));
            } else {
                panic!("Expected Binding");
            }
            assert!(matches!(nodes[1].kind, TypedExpressionKind::LocalVariable(VarKind::Local, 0)));
        } else {
            panic!("Expected Sequence");
        }
    }

}
