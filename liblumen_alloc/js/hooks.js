// This is based on wasm-tracing-allocator, an MIT/Apache 2.0 licensed crate
// which can be found at https://github.com/rustwasm/wasm-tracing-allocator
//
// We extend its functionality with support for tagged allocators, and therefore
// displaying live allocations/invalid frees on a per-allocator basis
(function () {
    class Allocator {
      constructor(tag) {
        this.tag = tag;
        this.liveAllocs = new Map();
        this.invalidFrees = new Map();
      }

      onAlloc(size, align, pointer) {
        this.liveAllocs.set(pointer, new Allocation(size, align, pointer));
      }

      onDealloc(size, align, pointer) {
        const wasLive = this.liveAllocs.delete(pointer);
        if (!wasLive) {
          this.invalidFrees.push(new InvalidFree(size, align, pointer));
        }
      }

      onRealloc(oldSize, newSize, align, oldPointer, newPointer) {
        this.onDealloc(oldSize, align, oldPointer);
        this.onAlloc(newSize, align, newPointer);
      }
    }

    class Allocation {
      constructor(size, align, pointer) {
        this.size = size;
        this.align = align;
        this.pointer = pointer;
        this.stack = getStack();
      }
    }
  
    class InvalidFree {
      constructor(size, align, pointer) {
        this.size = size;
        this.align = align;
        this.pointer = pointer;
        this.stack = getStack();
      }
    }
  
    const allocators = new Map();
  
    function getStack() {
      return Error()
        .stack
        .split("\n")
        .filter(frame => frame.match(/hooks\.js/) === null)
        .join("\n");
    }
  
    function onAlloc(tag, size, align, pointer) {
      let allocator = allocators.get(tag);
      allocator.onAlloc(size, align, pointer);
    }
  
    function onDealloc(tag, size, align, pointer) {
      let allocator = allocators.get(tag);
      allocator.onDealloc(size, align, pointer);
    }
  
    function onRealloc(
      tag,
      oldSize,
      newSize,
      align,
      oldPointer,
      newPointer,
    ) {
      let allocator = allocators.get(tag);
      allocator.onRealloc(oldSize, newSize, align, oldPointer, newPointer);
    }
  
    function dumpTable(entries, { keyLabel, valueLabel, getKey, getValue }) {
      const byKey = new Map;
      let total = 0;
  
      for (const entry of entries) {
        const key = getKey(entry);
        const keyValue = byKey.get(key) || 0;
        const entryValue = getValue(entry);
        total += entryValue;
        byKey.set(key, keyValue + entryValue);
      }
  
      const table = [...byKey]
            .sort((a, b) => b[1] - a[1])
            .map(a => ({ [keyLabel]: a[0], [valueLabel]: a[1] }));
  
      table.unshift({ [keyLabel]: "<total>", [valueLabel]: total });
  
      console.table(table, [keyLabel, valueLabel]);
    }

    function dumpAllocators(entries, { getValues, withValues }) {
      const byKey = new Map;
      let total = 0;

      for (const entry of entries) {
        const key = entry.tag;
        console.group(key);
        const values = getValues(entry);
        withValues(values);
        console.groupEnd(key);
      }
    }
  
    function getGlobal() {
      if (typeof self !== 'undefined') { return self; }
      if (typeof window !== 'undefined') { return window; }
      if (typeof global !== 'undefined') { return global; }
      throw new Error('unable to locate global object');
    }
  
    getGlobal().StatsAlloc = {
      on_alloc: onAlloc,
      on_dealloc: onDealloc,
      on_realloc: onRealloc,
  
      dumpLiveAllocations(opts) {
        dumpAllocators(allocators.values(), {
          getValues: entry => entry.liveValues.values(),
          withValues: live => dumpTable(live, Object.assign({
            keyLabel: "Live Allocations",
            valueLabel: "Size (Bytes)",
            getKey: entry => entry.stack,
            getValue: _entry => 1,
          }, opts)),
        });
      },
  
      dumpInvalidFrees(opts) {
        dumpAllocators(allocators.values(), {
          getValues: entry => entry.invalidFrees,
          withValues: invalid => dumpTable(invalid, Object.assign({
            keyLabel: "Invalid Free",
            valueLabel: "Count",
            getKey: entry => entry.stack,
            getValue: _entry => 1,
          }, opts))
        })
      },
    };
  }());
  