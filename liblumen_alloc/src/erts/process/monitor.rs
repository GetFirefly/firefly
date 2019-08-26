use crate::erts::term::{Atom, Pid};

pub enum Monitor {
    /// The monitor was created using a `Pid`, so the monitor message object should be the
    /// monitored `Process`'s `pid_term`
    Pid { monitoring_pid: Pid },
    /// When monitoring a name, it does not matter if the name change after the monitor, the name
    /// passed to monitor is always returned, but the node name reflects the current node name.
    ///
    /// ```elixir
    /// child = spawn_link(fn ->
    ///   receive do
    ///     {parent_pid, :change_name} ->
    ///       Process.unregister(:first_name)
    ///       Process.register(self(), :second_name)
    ///       send(parent_pid, :name_changed)
    ///   end
    ///
    ///   receive do
    ///     :exit ->
    ///       exit(:normal)
    ///   end
    /// end)
    ///
    /// Process.register(child, :first_name)
    /// :net_kernel.start([:first_node_name, :shortnames])
    /// :erlang.node()
    /// :erlang.monitor(:process, {:first_name, :erlang.node()})
    /// :net_kernel.stop()
    /// :net_kernel.start([:second_node_name, :shortnames])
    /// send(:first_name, {self(), :change_name})
    /// receive do
    ///   :name_changed ->
    ///     IO.puts("first_name is #{inspect(Process.whereis(:first_name))} while second_name is #{inspect(Process.whereis(:second_name))}")
    ///     send(child, :exit)
    ///     flush()
    /// end
    /// ```
    ///
    /// Ouputs:
    ///
    /// ```text
    /// first_name is nil while second_name is #PID<0.122.0>
    /// {:DOWN, #Reference<0.3637422028.3801874438.147163>, :process,
    ///   {:first_name, :second_node_name@Medina}, :normal}
    /// ```
    Name {
        monitoring_pid: Pid,
        monitored_name: Atom,
    },
}

impl Monitor {
    pub fn monitoring_pid(&self) -> &Pid {
        match self {
            Self::Pid { monitoring_pid } => monitoring_pid,
            Self::Name { monitoring_pid, .. } => monitoring_pid,
        }
    }
}
