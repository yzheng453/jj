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


use jj_lib::backend::ChangeId;
use jj_lib::commit::Commit;

use crate::eval_typechecker::TypedExpressionNode;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EvalType {
    Unit,
    Boolean,
    Integer,
    String,
    Commit,
    ChangeId,
    List(Box<EvalType>),
    Map(Box<EvalType>, Box<EvalType>),
    Lambda(Vec<EvalType>, Box<EvalType>),
    TypeVar(usize),
}

#[derive(Clone, Debug)]
pub enum Value<'i> {
    Unit,
    Boolean(bool),
    Integer(i64),
    String(std::borrow::Cow<'i, str>),
    Commit(Commit),
    ChangeId(ChangeId),
    List(Vec<Value<'i>>),
    Map(std::collections::HashMap<Value<'i>, Value<'i>>),
    // A closure that holds its number of parameters, the body, and its captured values
    Lambda(usize, TypedExpressionNode<'i>, Vec<Value<'i>>),
}

impl<'i> PartialEq for Value<'i> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Unit, Self::Unit) => true,
            (Self::Boolean(l0), Self::Boolean(r0)) => l0 == r0,
            (Self::Integer(l0), Self::Integer(r0)) => l0 == r0,
            (Self::String(l0), Self::String(r0)) => l0 == r0,
            (Self::Commit(l0), Self::Commit(r0)) => l0.id() == r0.id(),
            (Self::ChangeId(l0), Self::ChangeId(r0)) => l0 == r0,
            (Self::List(l0), Self::List(r0)) => l0 == r0,
            (Self::Map(l0), Self::Map(r0)) => l0 == r0,
            (Self::Lambda(_, _, _), Self::Lambda(_, _, _)) => false, // Lambdas are not comparable
            _ => false,
        }
    }
}

impl<'i> Eq for Value<'i> {}

impl<'i> std::hash::Hash for Value<'i> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Value::Unit => {}
            Value::Boolean(v) => v.hash(state),
            Value::Integer(v) => v.hash(state),
            Value::String(v) => v.hash(state),
            Value::Commit(v) => v.id().hash(state),
            Value::ChangeId(v) => v.hash(state),
            Value::List(v) => v.hash(state),
            Value::Map(_) => panic!("Cannot hash Map"),
            Value::Lambda(_, _, _) => panic!("Cannot hash Lambda"),
        }
    }
}

pub struct FunctionSignature {
    pub params: Vec<EvalType>,
    pub ret: EvalType,
    pub mutates: bool,
}
