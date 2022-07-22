use super::*;

/// Phase 4: Lower `receive` to more primitive operations
///
/// Lower a `receive` to more primitive operations. Rewrite patterns
/// that use and bind the same variable as nested cases.
///
/// Here follows an example of how a receive in this Erlang code:
///
/// foo(Timeout) ->
///     receive
///         {tag,Msg} -> Msg
///     after
///         Timeout ->
///             no_message
///     end.
///
/// is translated into Core Erlang:
///
/// 'foo'/1 =
///     fun (Timeout) ->
///         ( letrec
///               'recv$^0'/0 =
///                   fun () ->
///                       let <PeekSucceeded,Message> =
///                           primop 'recv_peek_message'()
///                       in  case PeekSucceeded of
///                             <'true'> when 'true' ->
///                                 case Message of
///                                   <{'tag',Msg}> when 'true' ->
///                                       do  primop 'remove_message'()
///                                           Msg
///                                   ( <Other> when 'true' ->
///                                         do  primop 'recv_next'()
///                                             apply 'recv$^0'/0()
///                                     -| ['compiler_generated'] )
///                                 end
///                             <'false'> when 'true' ->
///                                 let <TimedOut> =
///                                     primop 'recv_wait_timeout'(Timeout)
///                                 in  case TimedOut of
///                                       <'true'> when 'true' ->
///                                           'no_message'
///                                       <'false'> when 'true' ->
///                                           apply 'recv$^0'/0()
///                                     end
///                           end
///           in  apply 'recv$^0'/0()
///           -| ['letrec_goto'] )
pub struct SimplifyCst {
    context: Rc<UnsafeCell<FunctionContext>>,
}
impl SimplifyCst {
    pub(super) fn new(context: Rc<UnsafeCell<FunctionContext>>) -> Self {
        Self { context }
    }

    #[inline(always)]
    fn context(&self) -> &FunctionContext {
        unsafe { &*self.context.get() }
    }

    #[inline(always)]
    fn context_mut(&self) -> &mut FunctionContext {
        unsafe { &mut *self.context.get() }
    }
}
impl Pass for SimplifyCst {
    type Input<'a> = cst::Fun;
    type Output<'a> = cst::Fun;

    fn run<'a>(&mut self, _fun: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        todo!()
    }
}
