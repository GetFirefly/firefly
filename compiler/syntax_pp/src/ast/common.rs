use super::{expr, LineNum};

#[derive(Debug, Clone)]
pub struct Match<L, R> {
    pub line: LineNum,
    pub left: L,
    pub right: R,
}
impl_node!(Match<T,U>);
impl<L, R> Match<L, R> {
    pub fn new(line: LineNum, left: L, right: R) -> Self {
        Match { line, left, right }
    }
}
impl<L, R> Node for Match<L, R> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Tuple<T> {
    pub line: LineNum,
    pub elements: Vec<T>,
}
impl<T> Tuple<T> {
    pub fn new(line: LineNum, elements: Vec<T>) -> Self {
        Tuple { line, elements }
    }
}
impl<T> Node for Tuple<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Nil {
    pub line: LineNum,
}
impl Nil {
    pub fn new(line: LineNum) -> Self {
        Nil { line }
    }
}
impl Node for Nil {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Cons<T> {
    pub line: LineNum,
    pub head: T,
    pub tail: T,
}
impl<T> Cons<T> {
    pub fn new(line: LineNum, head: T, tail: T) -> Self {
        Cons { line, head, tail }
    }
}
impl<T> Node for Cons<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Binary<T> {
    pub line: LineNum,
    pub elements: Vec<BinElement<T>>,
}
impl<T> Binary<T> {
    pub fn new(line: LineNum, elements: Vec<BinElement<T>>) -> Self {
        Binary { line, elements }
    }
}
impl<T> Node for Binary<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct BinElement<T> {
    pub line: LineNum,
    pub element: T,
    pub size: Option<T>,
    pub tsl: Option<Vec<BinElementTypeSpec>>,
}
impl<T> BinElement<T> {
    pub fn new(line: LineNum, element: T) -> Self {
        BinElement {
            line,
            element,
            size: None,
            tsl: None,
        }
    }
    pub fn size(mut self, size: T) -> Self {
        self.size = Some(size);
        self
    }
    pub fn tsl(mut self, tsl: Vec<BinElementTypeSpec>) -> Self {
        self.tsl = Some(tsl);
        self
    }
}
impl<T> Node for BinElement<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct BinElementTypeSpec {
    pub name: String,
    pub value: Option<u64>,
}
impl BinElementTypeSpec {
    pub fn new(name: String, value: Option<u64>) -> Self {
        BinElementTypeSpec { name, value }
    }
}

#[derive(Debug, Clone)]
pub struct UnaryOp<T> {
    pub line: LineNum,
    pub operator: String,
    pub operand: T,
}
impl<T> UnaryOp<T> {
    pub fn new(line: LineNum, operator: String, operand: T) -> Self {
        UnaryOp {
            line,
            operator,
            operand,
        }
    }
}
impl<T> Node for UnaryOp<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct BinaryOp<T> {
    pub line: LineNum,
    pub operator: String,
    pub left_operand: T,
    pub right_operand: T,
}
impl<T> BinaryOp<T> {
    pub fn new(line: LineNum, operator: String, left_operand: T, right_operand: T) -> Self {
        BinaryOp {
            line,
            operator,
            left_operand,
            right_operand,
        }
    }
}
impl<T> Node for BinaryOp<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Record<T> {
    pub line: LineNum,
    pub base: Option<expr::Expression>,
    pub name: String,
    pub fields: Vec<RecordField<T>>,
}
impl<T> Record<T> {
    pub fn new(line: LineNum, name: String, fields: Vec<RecordField<T>>) -> Self {
        Record {
            line,
            base: None,
            name,
            fields,
        }
    }
    pub fn base(mut self, base: expr::Expression) -> Self {
        self.base = Some(base);
        self
    }
}
impl<T> Node for Record<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct RecordField<T> {
    pub line: LineNum,
    pub name: Option<String>, // `None` means `_` (i.e., default value)
    pub value: T,
}
impl<T> RecordField<T> {
    pub fn new(line: LineNum, name: Option<String>, value: T) -> Self {
        RecordField { line, name, value }
    }
}
impl<T> Node for RecordField<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct RecordIndex<T> {
    pub line: LineNum,
    pub base: Option<T>,
    pub record: String,
    pub field: String,
}
impl<T> RecordIndex<T> {
    pub fn new(line: LineNum, record: String, field: String) -> Self {
        RecordIndex {
            line,
            record,
            field,
            base: None,
        }
    }
    pub fn base(mut self, base: T) -> Self {
        self.base = Some(base);
        self
    }
}
impl<T> Node for RecordIndex<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Map<T> {
    pub line: LineNum,
    pub base: Option<expr::Expression>,
    pub pairs: Vec<MapPair<T>>,
}
impl<T> Map<T> {
    pub fn new(line: LineNum, pairs: Vec<MapPair<T>>) -> Self {
        Map {
            line,
            base: None,
            pairs,
        }
    }
    pub fn base(mut self, base: expr::Expression) -> Self {
        self.base = Some(base);
        self
    }
}
impl<T> Node for Map<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct MapPair<T> {
    pub line: LineNum,
    pub is_assoc: bool,
    pub key: T,
    pub value: T,
}
impl<T> MapPair<T> {
    pub fn new(line: LineNum, is_assoc: bool, key: T, value: T) -> Self {
        MapPair {
            line,
            is_assoc,
            key,
            value,
        }
    }
}
impl<T> Node for MapPair<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct LocalCall<T> {
    pub line: LineNum,
    pub function: T,
    pub args: Vec<T>,
}
impl<T> LocalCall<T> {
    pub fn new(line: LineNum, function: T, args: Vec<T>) -> Self {
        LocalCall {
            line,
            function,
            args,
        }
    }
}
impl<T> Node for LocalCall<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct RemoteCall<T> {
    pub line: LineNum,
    pub module: T,
    pub function: T,
    pub args: Vec<T>,
}
impl<T> RemoteCall<T> {
    pub fn new(line: LineNum, module: T, function: T, args: Vec<T>) -> Self {
        RemoteCall {
            line,
            module,
            function,
            args,
        }
    }
}
impl<T> Node for RemoteCall<T> {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct InternalFun {
    pub line: LineNum,
    pub function: String,
    pub arity: Arity,
}
impl InternalFun {
    pub fn new(line: LineNum, function: String, arity: Arity) -> Self {
        InternalFun {
            line,
            function,
            arity,
        }
    }
}
impl Node for InternalFun {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct ExternalFun {
    pub line: LineNum,
    pub module: expr::Expression,
    pub function: expr::Expression,
    pub arity: expr::Expression,
}
impl ExternalFun {
    pub fn new(
        line: LineNum,
        module: expr::Expression,
        function: expr::Expression,
        arity: expr::Expression,
    ) -> Self {
        ExternalFun {
            line,
            module,
            function,
            arity,
        }
    }
}
impl Node for ExternalFun {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Var {
    pub line: LineNum,
    pub name: String,
}
impl Var {
    pub fn new(line: LineNum, name: String) -> Self {
        Var { line, name }
    }
    pub fn is_anonymous(&self) -> bool {
        self.name == "_"
    }
}
impl Node for Var {
    fn line(&self) -> LineNum {
        self.line
    }
}
