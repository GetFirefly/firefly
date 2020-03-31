///! This module is just a sketch in Rust syntax of the algorithm, the code is not valie
pub trait GarbageCollector {
    fn gc(&mut self);
}

enum Cycle {
    None,
    Major,
    Minor,
}

/// During a collection cycle, the collector might exceed its allowable (time or work based) quantum, and
/// be forced to abort collection and yield to the mutator.
///
/// In incremental tracing collectors like this one, the amount of work to be done in one collection cycle
/// depends on the amount of live data when the root set is taken. Since we don't know the quantity, we
/// devise a mechanism that allows us to control how much allocation the mutator is allowed to do between stages.
///
/// When the mutator reaches the allocation limit, the collector is invoked.
///
/// # Design Notes
///
/// * Forwarding area implemented as a array of constant size, same size as the from-space,
/// * Preallocate an array of pages, 32k words per page, those not used by the old generation are kept in a linked list on the side
/// * Require locking the PCB when snapshotting roots; and when removing a process from the dirty set to prevent removing the dirty
///   bit from a process at the same time it becomes dirty by a message send
/// * Forwarding pointers are stored in a separate areas to allow the mutator to access objects in the from space between collection stages
///
/// ## Atomic Operations
///
/// Atomic opertions in the collector are stages during which the collector cannot be interuptted:
///
/// * Swapping the nursery and the from space (this has to be truly atomic, as mutators cannot be trying to allocate during the swap)
/// * Setting up and cleaning up auxiliary areas for collection, since checks for quantum expiration occur at the beginning of
///   various loops, anything internal occuring between checks are effectively atomic operations for the collector (they are only visible by the collector)
///
/// ## Optimizations
///
/// * Processes added to the dirty set at the beginning of a major collection are done so in order of age,
///   but each time a process receives a message, it is moved to the last in the set, as if it was a newly spawned process, this keeps the busiest
///   processes last, scanning them late as possible, making the dirty set effectively a queue. The reasons for this are:
///     * avoid repeated re-activation of message-exchanging processes
///     * allow processes to execute long enough to generate garbage
///     * give processes a chance to die before taking a snapshot of their root set, potentially allowing us to avoid considering them at all
/// * Process the stack of gray objects after each process, postponing the processing of other processes (as above)
/// * In minor collections, we remember the top of the heap for each process, and only consider roots data which was created since the process
///   was taken off the dirty process set
/// * When the collector rescues objects from the young gen to the old, it uses the free list; but since a new page is allocated at the beginning
///   of each major collection, we can cheaply allocate in this page by pointer bumping during the collection
/// * A key optimization is to have process-local collections record pointers into the message area in a remembered set; this way we avoid scanning
///   the old generation of their local heaps.
pub struct Incremental {
    cycle: Cycle,
    next_cycle: Option<Cycle>,
    nursery: *const u8,
    from_space: *const u8,
}
impl GarbageCollector for Incremental {
    fn gc(&mut self) {
        match self.cycle {
            // Set if we're in a cycle and it is a major
            Gc::Major => self.major_gc(),
            // Set if we're in a cycle, but not major
            Gc::Minor => self.minor_gc(),
            // Set if we're not in a cycle
            // Indicates we need to start a new cycle
            Gc::Idle => {
                // Start by swapping the nursery and from space roles, this is atomic
                self.swap();
                // Resets all the forwarding pointers
                self.clear_forwarding_area();
                // If during the last cycle we determined a major collection was needed, this is set
                match self.next_cycle {
                    Some(Gc::Major) => {
                        self.cycle = Gc::Major;
                        self.next_cycle = None;
                        self.gc();
                    },
                    _ => {
                        self.cycle = Gc::Minor;
                        self.gc();
                    }
                }
            }
        };
        // Calculates how much the mutator is allowed to work before next collection
        self.update_allocation_limit();
    }

    // Incremental minor collection.
    // All roots are traversed and live objects in the from space are copied to
    // the old generation and marked as gray. Afterwards, all gray objects are traversed
    // in a similar way to copy their children
    //
    // Starts by picking up the first process from the dirty set and conceptually takes
    // a snapshot of its root set. We do not want to take an actual copy, so we record
    // the values of the stack and process-local heap pointer and the pointers in the message queue;
    // in other words, taking a snapshot consists of recording a set of pointers.
    //
    // Afterwards, the mutator can continue allocating in the nursery and reading from the from space
    // as there are no destructive updates. The snapshot is scanned and when a live object is found
    // in the from space and this object has not yet been forwarded, the object is copied to the old
    // generation and added to a stack of gray objects. Each time we copy an object, we update the original
    // root references to point to the new location (update source ref), and store a forwarding pointer
    // in the forward area to ensure objects are copied at most once (set foward pointer). If the object
    // was previously forwarded, we just update its reference in the root set
    //
    // When all roots are scanned, we pop the gray objects one by one and each object is scanned for references
    // If the popped object refers to a an object in the from space that has not already been forwarded, the
    // newly found object is copied and pushed onto the gray stack. In the generational setting, an object is
    // gray if it has been copied to the old generation but not yet scanned for references to other objects. An
    // object is fully processed and becomes black when all its children are either black or gray
    //
    // Eventually the gray stack is empty and the process is removed from the dirty set, and we move on to the next process.
    //
    // During a collection cycle, processes may become dirty again only by receiving a message allocated in the from space,
    // in effect this acts as a write barrier, as it requires one extra test for each send operation.
    //
    // At the end of the cycle, we also have to look through the objects in the nursery to update references still
    // pointing to the from space (or possibly copy the referred objects), since the mutator can create references from
    // objects in the nursery to objects in the from space between collection stages. Because the nursery might contain
    // references to objects not copied to the old generation yet, a final check that the gray stack is empty is needed at the end
    fn minor_gc(&mut self) {
        for p in self.dirty_process_set {
            let roots = self.collect_roots(p); // atomic in time-based collector
            for r in roots {
                if self.quantum_expired {
                    return;
                }
                if self.points_to(r, self.from_space) {
                    self.forward(r);
                }
            }

            for g in self.gray() {
                for rf in g.reference_fields() {
                    if self.quantum_expired {
                        return;
                    }
                    if self.points_to(rf, self.from_space) {
                        self.forward(rf);
                    }
                }
                self.mark_black(g); // remove g from the gray stack
            }
            self.remove_from_dirty_process_set(p);
        }

        for r in self.nursery.references() {
            if self.quantum_expired {
                return;
            }
            if self.points_to(r, self.from_space) {
                self.forward(r);
            }
        }

        for g in self.gray() {
            for rf in g.reference_fields() {
                if self.quantum_expired {
                    return;
                }
                if self.points_to(rf, self.from_space) {
                    self.forward(rf);
                }
            }
            self.mark_black(g); // remove g from the gray stack
        }

        self.cycle = Cycle::None;
    }

    // Incremental major collection
    // Marks all processes as dirty, collects old generation
    //
    // A new page is linked to the old generation and added to the free list to allow
    // the copying collector of the young generation to finish its current minor collection
    //
    // The major collector is a combination of a copying collector and a mark/sweep collector,
    // the copying part is the same as in the minor collection, copying objects from the young to the old
    // generation, linking in new pages to the old generation as needed; the old generation itself is collected
    // by the mark/sweep collector.
    //
    // While scanning the root set, reachable objects in the young generation that are not already marked are copied
    // to the old generation, pushed onto the gray stak and immediately marked as live; reachable objects in the old
    // generation are the same, except no copying is needed.
    //
    // When all roots have been scanned, like in the minor collection, we pop all the gray objects; if the popped object
    // refers to an object that has not already been marked as live, the referred object is copied if it resides in the
    // young generation, pushed onto the gray stack and marked
    //
    // Because the mutator (which allocates in the nursery) might, in the meantime, have created pointers to the message
    // area, we also scan the nursery. When all dirty processes have been processed and we are out of gray objects, we proceed
    // to the sweep phase. In the sweep phase, we simply scan the black map and build a new free-list from the unmarke areas.
    // If an entire page turns out to be free, we release the page.
    fn major_gc(&mut self) {
        match self.next_cycle {
            Some(Gc::Major) => {
                self.mark_all_as_dirty();
                self.clear_blackmap();
                self.cycle = Gc::Major;
                self.next_cycle = None;
                self.fetch_new_page_for_old();
            },
            _ => ()
        }

        for p in self.dirty_process_set {
            let roots = self.collect_roots(p);
            for r in roots {
                if self.quantum_expired {
                    return
                }
                self.copymark(r);
            }

            for g in self.gray() {
                for rf in g.reference_fields() {
                    if self.quantum_expired {
                        return
                    }
                    self.copymark(rf);
                }
                self.mark_black(g);
            }
            self.remove_from_dirty_process_set(p)
        }

        for r in self.nursery.references() {
            if self.quantum_expired {
                return;
            }
            self.copymark(r);
        }

        for g in self.gray() {
            for rf in g.reference_fields() {
                if self.quantum_expired {
                    return
                }
                self.copymark(rf);
            }
            self.mark_black(g);
        }

        self.sweep(); // traverse the black map and build a free-list of the unmarked areas
        self.cycle = Cycle::None;
    }

    fn swap(&mut self) {
        let swap = self.nursery;
        self.nursery = self.from_space;
        self.from_space = swap;
    }

    fn clear_forwarding_area(&mut self) {
    }

    fn update_allocation_limit(&mut self) {
    }

    fn copymark(&mut self, r) {
        if self.points_to(r, self.from_space) {
            self.forward(r);
            self.mark_blackmap(r);
        } else if self.points_to(r, self.old_generation) {
            match self.color(r) {
                Tricolor::White => {
                    self.push_gray(r);
                    self.mark_blackmap(r);
                },
                _ => ()
            }
        }
    }

    // Copies objects to the old generation
    // If allocation in the old generation fails, we set the next cycle to Major
    fn copy_object(&self, obj) -> copy {
        if self.available_in_old < sizeof(obj) {
            self.allocate_new_page();
            match self.cycle {
                Cycle::Major => (),
                _ => self.next_cycle = Some(Cycle::Major)
            }
        }
        self.copy_to_old(obj)
    }

    // Forwards objects to the old generation, if not already forwarded, it is
    // copied to the old generation and pushed onto the gray stack
    fn forward(&mut self, obj) {
        if self.is_forwarded(obj) {
            self.update_source_ref(obj);
        } else {
            let destination = self.copy_object(obj);
            self.push_gray(destination.clone());
            self.update_source_ref(obj);
            self.set_fowarding_pointer(obj, destination);
        }
    }
}
