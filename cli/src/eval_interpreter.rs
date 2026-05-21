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


use thiserror::Error;

use jj_lib::repo::{MutableRepo, ReadonlyRepo};

use crate::eval_typechecker::{TypedExpressionKind, TypedExpressionNode};
use crate::eval_types::Value;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("Read-only transaction: {0}")]
    ReadOnlyTransaction(&'static str),
    #[error("Evaluation error: {0}")]
    GenericError(String),
}

pub enum RepoContext<'a> {
    ReadOnly(&'a ReadonlyRepo),
    Mutable(&'a mut MutableRepo),
    #[cfg(test)]
    DummyForTests,
}

pub struct EvalContext<'a, 'i> {
    pub repo: RepoContext<'a>,
    pub stack: Vec<Value<'i>>,
    pub fp: usize,
}

pub fn call_lambda<'a, 'i>(
    lambda_val: &Value<'i>,
    args: Vec<Value<'i>>,
    ctx: &mut EvalContext<'a, 'i>,
) -> Result<Value<'i>, EvalError> {
    let Value::Lambda(num_args, body, captures) = lambda_val else {
        return Err(EvalError::GenericError("Not a lambda".to_string()));
    };
    if args.len() != *num_args {
        return Err(EvalError::GenericError("Arity mismatch".to_string()));
    }
    
    let old_fp = ctx.fp;
    ctx.fp = ctx.stack.len();
    
    for c in captures {
        ctx.stack.push(c.clone());
    }
    for a in args {
        ctx.stack.push(a);
    }
    
    let res = evaluate(body, ctx);
    
    ctx.stack.truncate(ctx.fp);
    ctx.fp = old_fp;
    
    res
}

pub fn evaluate<'a, 'i>(
    node: &TypedExpressionNode<'i>,
    ctx: &mut EvalContext<'a, 'i>,
) -> Result<Value<'i>, EvalError> {
    match &node.kind {
        TypedExpressionKind::Integer(i) => Ok(Value::Integer(*i)),
        TypedExpressionKind::Boolean(b) => Ok(Value::Boolean(*b)),
        TypedExpressionKind::String(s) => Ok(Value::String(s.clone())),
        TypedExpressionKind::LocalVariable(_, index) => Ok(ctx.stack[ctx.fp + *index].clone()),
        TypedExpressionKind::Binding(slot, value_node) => {
            let val = evaluate(value_node, ctx)?;
            assert_eq!(ctx.stack.len(), ctx.fp + *slot);
            ctx.stack.push(val);
            Ok(Value::Unit)
        }
        TypedExpressionKind::Sequence(nodes) => {
            let mut result = Value::Unit;
            let start_len = ctx.stack.len();
            for node in nodes {
                result = evaluate(node, ctx)?;
            }
            ctx.stack.truncate(start_len);
            Ok(result)
        }
        TypedExpressionKind::Lambda(lambda) => {
            let mut captures = Vec::new();
            for capture in &lambda.captures {
                captures.push(ctx.stack[ctx.fp + capture.outer_index].clone());
            }
            Ok(Value::Lambda(lambda.num_args, lambda.body.clone(), captures))
        }
        TypedExpressionKind::FunctionCall(call) => {
            if call.function_name == "abandon" {
                let commit_val = evaluate(&call.args[0], ctx)?;
                let RepoContext::Mutable(mut_repo) = &mut ctx.repo else {
                    return Err(EvalError::ReadOnlyTransaction(
                        "Cannot call abandon() in read-only mode",
                    ));
                };

                if let Value::Commit(c) = commit_val {
                    mut_repo.record_abandoned_commit(&c);
                }
                return Ok(Value::Unit);
            }
            
            if call.function_name == "map_int" || call.function_name == "map_bool" {
                let val = evaluate(&call.args[0], ctx)?;
                let lambda_val = evaluate(&call.args[1], ctx)?;
                return call_lambda(&lambda_val, vec![val], ctx);
            }
            
            if call.function_name == "group_by" {
                let list_val = evaluate(&call.args[0], ctx)?;
                let lambda_val = evaluate(&call.args[1], ctx)?;
                
                let Value::List(items) = list_val else {
                    return Err(EvalError::GenericError("group_by called on non-list".to_string()));
                };
                
                let mut map: std::collections::HashMap<Value<'i>, Value<'i>> = std::collections::HashMap::new();
                for item in items {
                    let key = call_lambda(&lambda_val, vec![item.clone()], ctx)?;
                    match map.get_mut(&key) {
                        Some(Value::List(group)) => {
                            group.push(item);
                        }
                        None => {
                            map.insert(key, Value::List(vec![item]));
                        }
                        _ => unreachable!(),
                    }
                }
                return Ok(Value::Map(map));
            }
            
            if call.function_name == "for_each" {
                let map_val = evaluate(&call.args[0], ctx)?;
                let lambda_val = evaluate(&call.args[1], ctx)?;
                
                let Value::Map(map) = map_val else {
                    return Err(EvalError::GenericError("for_each called on non-map".to_string()));
                };
                
                for (k, v) in map {
                    call_lambda(&lambda_val, vec![k, v], ctx)?;
                }
                return Ok(Value::Unit);
            }

            // Add actual logic here
            todo!("Evaluate other functions")
        }
        TypedExpressionKind::Unary(op, arg) => {
            let val = evaluate(arg, ctx)?;
            match (op, val) {
                (crate::eval::UnaryOp::LogicalNot, Value::Boolean(b)) => Ok(Value::Boolean(!b)),
                (crate::eval::UnaryOp::Negate, Value::Integer(i)) => Ok(Value::Integer(-i)),
                _ => Err(EvalError::GenericError("Invalid unary operation".to_owned())),
            }
        }
        TypedExpressionKind::Binary(op, lhs, rhs) => {
            match op {
                crate::eval::BinaryOp::LogicalAnd => {
                    let left = evaluate(lhs, ctx)?;
                    if let Value::Boolean(false) = left {
                        return Ok(Value::Boolean(false));
                    }
                    return evaluate(rhs, ctx);
                }
                crate::eval::BinaryOp::LogicalOr => {
                    let left = evaluate(lhs, ctx)?;
                    if let Value::Boolean(true) = left {
                        return Ok(Value::Boolean(true));
                    }
                    return evaluate(rhs, ctx);
                }
                _ => {}
            }
            
            let left = evaluate(lhs, ctx)?;
            let right = evaluate(rhs, ctx)?;
            match (op, left, right) {
                (crate::eval::BinaryOp::LogicalAnd, Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(l && r)),
                (crate::eval::BinaryOp::LogicalOr, Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(l || r)),
                (crate::eval::BinaryOp::Eq, l, r) => Ok(Value::Boolean(l == r)),
                (crate::eval::BinaryOp::Ne, l, r) => Ok(Value::Boolean(l != r)),
                (crate::eval::BinaryOp::Gt, Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l > r)),
                (crate::eval::BinaryOp::Ge, Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l >= r)),
                (crate::eval::BinaryOp::Lt, Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l < r)),
                (crate::eval::BinaryOp::Le, Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l <= r)),
                (crate::eval::BinaryOp::Add, Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l + r)),
                (crate::eval::BinaryOp::Sub, Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l - r)),
                (crate::eval::BinaryOp::Mul, Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l * r)),
                (crate::eval::BinaryOp::Div, Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l / r)),
                (crate::eval::BinaryOp::Rem, Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l % r)),
                _ => Err(EvalError::GenericError("Invalid binary operation".to_owned())),
            }
        }
        TypedExpressionKind::MethodCall(_) => unreachable!("MethodCall should be desugared to FunctionCall in type_check"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_typechecker::{TypeContext, type_check};

    #[test]
    fn test_nested_closures() {
        let mut type_ctx = TypeContext::default_builtins();
        let text = "map_int(5, |x| map_int(x, |y| x + y))";
        let raw_ast = crate::eval::parse_eval_expressions(text).unwrap();
        let typed_ast = type_check(&raw_ast, &mut type_ctx).unwrap();
        
        let mut eval_ctx = EvalContext {
            repo: RepoContext::DummyForTests,
            stack: Vec::new(),
            fp: 0,
        };
        
        let result = evaluate(&typed_ast, &mut eval_ctx).unwrap();
        assert_eq!(result, Value::Integer(10));
    }

    #[test]
    fn test_pipeline_evaluation() {
        let mut type_ctx = TypeContext::default_builtins();
        let text = "map_bool(15, |x| map_bool(x, |y| y > 10))";
        let raw_ast = crate::eval::parse_eval_expressions(text).unwrap();
        let typed_ast = type_check(&raw_ast, &mut type_ctx).unwrap();
        
        let mut eval_ctx = EvalContext {
            repo: RepoContext::DummyForTests,
            stack: Vec::new(),
            fp: 0,
        };
        
        let result = evaluate(&typed_ast, &mut eval_ctx).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }
}
