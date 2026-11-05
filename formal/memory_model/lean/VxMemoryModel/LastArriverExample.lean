import VxMemoryModel.ScopedCheck

/-!
# The last block of two, with device scope and with block scope

Two blocks run the code of `LastArriver.lean`. Block 1 adds last: it reads 1 from the add of
block 0, and then reads the partial result of block 0. With adds of device scope, the model allows
this run and it has no data race. With adds of block scope, the read of the partial races with
its write.
-/

namespace Vx

/-- Event 0 sets the counter (place 0) to 0. Block 0: 1 writes its partial (place 1), 2 is a
release fence, 3 and 4 add 1 to the counter. Block 1: 5 and 6 add 1, 7 is an acquire fence, 8
reads the partial of block 0. The adds have the scope `s`. -/
def lastOfTwo (s : Scope) : SData 9 := SData.ofLists 9
  [.write, .write, .fence, .read, .write, .read, .write, .fence, .read]
  [.plain, .plain, .rel, .rlx, .rlx, .rlx, .rlx, .acq, .plain]
  [0, 1, 2, 0, 0, 0, 0, 3, 1]
  [1000, 0, 0, 0, 0, 1, 1, 1, 1]
  ([1, 2, 3, 4, 5, 6, 7, 8].map (0, ·) ++
    [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8),
      (7, 8)])
  [(0, 3), (4, 5), (1, 8)] [(0, 4), (0, 6), (4, 6)] [(3, 4), (5, 6)]
  [.system, .system, .device, s, s, s, s, .device, .system]
  [(0, 0, 0), (1, 1, 0)] []

/-- With adds of device scope, the run is allowed and has no data race. -/
theorem lastOfTwo_device : (lastOfTwo .device).check = some (true, false) := by decide +kernel

/-- With adds of block scope, the run is allowed, and the write and the read of the partial
race. -/
theorem lastOfTwo_block : (lastOfTwo .block).check = some (true, true) ∧
    (lastOfTwo .block).raceOn 1 8 = some true := by decide +kernel

end Vx
