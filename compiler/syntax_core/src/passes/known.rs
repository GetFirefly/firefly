use rpds::{RedBlackTreeSet, Vector};

use firefly_intern::Ident;

/// Here follows an abstract data structure to help us handle Erlang's
/// implicit matching that occurs when a variable is bound more than
/// once:
///
///     X = Expr1(),
///     X = Expr2()
///
/// What is implicit in Erlang, must be explicit in Core Erlang; that
/// is, repeated variables must be eliminated and explicit matching
/// must be added. For simplicity, examples that follow will be given
/// in Erlang and not in Core Erlang. Here is how the example can be
/// rewritten in Erlang to eliminate the repeated variable:
///
///     X = Expr1(),
///     X1 = Expr2(),
///     if
///         X1 =:= X -> X;
///         true -> error({badmatch,X1})
///     end
///
/// To implement the renaming, keeping a set of the variables that
/// have been bound so far is **almost** sufficient. When a variable
/// in the set is bound a again, it will be renamed and a `case` with
/// guard test will be added.
///
/// Here is another example:
///
///     (X=A) + (X=B)
///
/// Note that the operands for a binary operands are allowed to be
/// evaluated in any order. Therefore, variables bound on the left
/// hand side must not referenced on the right hand side, and vice
/// versa. If a variable is bound on both sides, it must be bound
/// to the same value.
///
/// Using the simple scheme of keeping track of known variables,
/// the example can be rewritten like this:
///
///     X = A,
///     X1 = B,
///     if
///         X1 =:= X -> ok;
///         true -> error({badmatch,X1})
///     end,
///     X + X1
///
/// However, this simple scheme of keeping all previously bound variables in
/// a set breaks down for this example:
///
///     (X=A) + fun() -> X = B end()
///
/// The rewritten code would be:
///
///     X = A,
///     Tmp = fun() ->
///               X1 = B,
///               if
///                   X1 =:= X -> ok;
///                   true -> error({badmatch,X1})
///               end
///           end(),
///     X + Tmp
///
/// That is wrong, because the binding of `X` created on the left hand
/// side of `+` must not be seen inside the fun. The correct rewrite
/// would be like this:
///
///     X = A,
///     Tmp = fun() ->
///               X1 = B
///           end(),
///     X + Tmp
///
/// To correctly rewrite fun bodies, we will need to keep addtional
/// information in a record so that we can remove `X` from the known
/// variables when rewriting the body of the fun.
///
#[derive(Debug, Clone, Default)]
pub struct Known {
    base: Vector<RedBlackTreeSet<Ident>>,
    ks: RedBlackTreeSet<Ident>,
    prev_ks: Vector<RedBlackTreeSet<Ident>>,
}
impl Known {
    /// Get the currently known variables
    pub fn get(&self) -> &RedBlackTreeSet<Ident> {
        &self.ks
    }

    /// Returns true if the given ident is known in the current scope
    #[inline]
    pub fn contains(&self, id: &Ident) -> bool {
        self.ks.contains(id)
    }

    /// Returns true if the known set is empty
    pub fn is_empty(&self) -> bool {
        self.ks.is_empty()
    }

    pub fn start_group(&mut self) {
        self.prev_ks.push_back_mut(RedBlackTreeSet::new());
        self.base.push_back_mut(self.ks.clone());
    }

    pub fn end_body(&mut self) {
        self.prev_ks.drop_last_mut();
        self.prev_ks.push_back_mut(self.ks.clone());
    }

    /// Consolidate the known variables after having processed the
    /// last body in a group of bodies that see the same bindings.
    pub fn end_group(&mut self) {
        self.base.drop_last_mut();
        self.prev_ks.drop_last_mut();
    }

    /// Update the known variables to be the union of the previous
    /// known variables and the set KnownVarsSet.
    pub fn union(&self, vars: &RedBlackTreeSet<Ident>) -> Self {
        let ks = vars
            .iter()
            .copied()
            .fold(self.ks.clone(), |ks, var| ks.insert(var));
        Self {
            base: self.base.clone(),
            ks,
            prev_ks: self.prev_ks.clone(),
        }
    }

    /// Add variables that are known to be bound in the current body.
    pub fn bind(&self, vars: &RedBlackTreeSet<Ident>) -> Self {
        let last = self.prev_ks.last().map(|set| set.clone());
        let prev_ks = self.prev_ks.drop_last();
        match last {
            None => self.clone(),
            Some(mut last) => {
                let mut prev_ks = prev_ks.unwrap();
                // set difference of prev_ks and vars
                for v in vars.iter() {
                    last.remove_mut(v);
                }
                prev_ks.push_back_mut(last);
                Self {
                    base: self.base.clone(),
                    ks: self.ks.clone(),
                    prev_ks,
                }
            }
        }
    }

    /// Update the known variables to only the set of variables that
    /// should be known when entering the fun.
    pub fn known_in_fun(&self, name: Option<Ident>) -> Self {
        if self.base.is_empty() || self.prev_ks.is_empty() {
            if let Some(name) = name {
                let mut ks = self.ks.clone();
                ks.insert_mut(name);
                return Self {
                    base: self.base.clone(),
                    ks,
                    prev_ks: self.prev_ks.clone(),
                };
            } else {
                return self.clone();
            }
        }

        // Within a group of bodies that see the same bindings, calculate
        // the known variables for a fun. Example:
        //
        //     A = 1,
        //     {X = 2, fun() -> X = 99, A = 1 end()}.
        //
        // In this example:
        //
        //     BaseKs = ['A'], Ks0 = ['A','X'], PrevKs = ['A','X']
        //
        // Thus, only `A` is known when entering the fun.
        let mut ks = self.ks.clone();
        let prev_ks = self.prev_ks.last().map(|l| l.clone()).unwrap_or_default();
        let base = self.base.last().map(|l| l.clone()).unwrap_or_default();
        for id in prev_ks.iter() {
            ks.remove_mut(id);
        }
        for id in base.iter() {
            ks.insert_mut(*id);
        }
        if let Some(name) = name {
            ks.insert_mut(name);
        }
        Self {
            base: Vector::new(),
            prev_ks: Vector::new(),
            ks,
        }
    }
}
