use std::future::Future;
use std::panic::{resume_unwind, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread;

use crossbeam::channel;

use futures::executor;
use futures::future::FutureExt;

use lazy_static::lazy_static;

/// Spawns a future on the thread pool
///
/// The returned handle can be used to await the output of the future with `join`
pub fn spawn<F, R>(future: F) -> JoinHandle<R>
where
    F: Future<Output = R> + Send + 'static,
    R: Send + 'static,
{
    lazy_static! {
        static ref SCHEDULER: Scheduler = Scheduler::new(num_cpus::get());
    }

    SCHEDULER.spawn(future)
}

/// Joins a task to the current thread by blocking until completion.
#[inline(always)]
pub fn join<R>(handle: JoinHandle<R>) -> Option<R>
where
    R: Send + 'static,
{
    executor::block_on(handle)
}

enum Message {
    Run(async_task::Task<()>),
    Close,
}

struct Scheduler {
    size: usize,
    threads: Arc<Vec<ThreadState>>,
}

struct ThreadState {
    tx: Arc<channel::Sender<Message>>,
}

impl Scheduler {
    pub fn new(size: usize) -> Self {
        let mut threads = Vec::with_capacity(size);
        for _ in 0..size {
            let (tx, rx) = channel::unbounded();

            thread::spawn(move || {
                let _scope = executor::enter().unwrap();
                // Keep taking the task from the channel and running it until completion.
                for message in rx {
                    match message {
                        Message::Run(task) => task.run(),
                        Message::Close => return,
                    }
                }
            });

            threads.push(ThreadState { tx: Arc::new(tx) });
        }

        Self {
            size,
            threads: Arc::new(threads),
        }
    }

    /// Spawns a future on the thread pool
    ///
    /// The returned handle can be used to await the output of the future with `join`
    pub fn spawn<F, R>(&self, future: F) -> JoinHandle<R>
    where
        F: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        use rand::distributions::{Distribution, Uniform};

        // Select thread
        let range = Uniform::from(0..self.size);
        let mut rng = rand::thread_rng();
        let index = range.sample(&mut rng);
        let thread = &self.threads[index];

        // Wrap the future into one that disconnects the channel on completion.
        let future = AssertUnwindSafe(future).catch_unwind();

        // Create a task that is scheduled by sending itself into the channel.
        let tx = Arc::downgrade(&thread.tx);
        let schedule = move |t| tx.upgrade().unwrap().send(Message::Run(t)).unwrap();
        let (task, handle) = async_task::spawn(future, schedule, ());

        // Schedule the task by sending it into the channel.
        task.schedule();

        // Wrap the handle in one that propagates panics
        JoinHandle(handle)
    }
}
impl Drop for Scheduler {
    fn drop(&mut self) {
        for thread in self.threads.iter() {
            thread.tx.send(Message::Close).unwrap();
        }
    }
}

/// A join handle that propagates panics inside the task.
pub struct JoinHandle<R>(async_task::JoinHandle<thread::Result<R>, ()>);
impl<R> Future for JoinHandle<R> {
    type Output = Option<R>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.0).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Ready(Some(Ok(val))) => Poll::Ready(Some(val)),
            Poll::Ready(Some(Err(err))) => resume_unwind(err),
        }
    }
}
