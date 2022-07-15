use cranelift_entity::entity_impl;
use intrusive_collections::linked_list::{Cursor, LinkedList};
use intrusive_collections::{LinkedListLink, UnsafeRef};

use super::*;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Block(u32);
entity_impl!(Block, "block");
impl Default for Block {
    #[inline]
    fn default() -> Self {
        use cranelift_entity::packed_option::ReservedValue;

        Self::reserved_value()
    }
}

pub struct BlockData {
    pub link: LinkedListLink,
    pub params: ValueList,
    pub insts: LinkedList<InstAdapter>,
}
impl Drop for BlockData {
    fn drop(&mut self) {
        self.insts.fast_clear();
    }
}
impl Clone for BlockData {
    fn clone(&self) -> Self {
        Self {
            link: LinkedListLink::default(),
            params: self.params.clone(),
            insts: LinkedList::new(InstAdapter::new()),
        }
    }
}
impl BlockData {
    pub(crate) fn new() -> Self {
        Self {
            link: LinkedListLink::default(),
            params: ValueList::new(),
            insts: LinkedList::new(InstAdapter::new()),
        }
    }

    pub fn insts<'f>(&'f self) -> impl Iterator<Item = Inst> + 'f {
        Insts {
            cursor: self.insts.front(),
        }
    }

    pub unsafe fn append(&mut self, inst: UnsafeRef<InstNode>) {
        self.insts.push_back(inst);
    }

    pub fn first(&self) -> Option<Inst> {
        self.insts.front().get().map(|data| data.key)
    }

    pub fn last(&self) -> Option<Inst> {
        self.insts.back().get().map(|data| data.key)
    }
}

struct Insts<'f> {
    cursor: Cursor<'f, InstAdapter>,
}
impl<'f> Iterator for Insts<'f> {
    type Item = Inst;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor.is_null() {
            return None;
        }
        let next = self.cursor.get().map(|data| data.key);
        self.cursor.move_next();
        next
    }
}
