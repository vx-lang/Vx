import VxMemoryModel.Check

/-!
# The orders of a broadcast channel, in small runs

Each run below has the places 0 (the value in a slot), 1 (the count of receivers that did not
read it) and 2 (the number of the slot). A `send` writes the value, sets the count, and sets the
number with a release store. Each receiver reads the number with an acquire load, as in
`stdlib/std/sync/broadcast.vx`. Each run is allowed by the model. With the orders of the library
it has no data race. With one weaker order, the clone of the value races with a later write of
it. `BroadcastOrders.lean` proves the same edges for every run.
-/

namespace Vx

/-- Receiver A clones the value and takes 1 from the count with the read-modify-write `o`.
Receiver B reads the count 1 with the load `l`, stores 0, moves the value out, and frees the
slot. Then the next `send` writes the slot.

Thread 0, the `send`: 0 writes the value, 1 sets the count to 2, 2 sets the number. Thread 1,
receiver A: 3 reads the number, 4 clones the value, 5 and 6 take 1 from the count. Thread 2,
receiver B: 7 reads the number, 8 reads the count, 9 stores 0, 10 moves the value out, 11 frees
the slot. Thread 3, the next `send`: 12 reads the number, 13 writes the value. -/
def cloneThenMove (o l : Order) : Data 14 := Data.ofLists 14
  [.write, .write, .write, .read, .read, .read, .write, .read, .read, .write, .read, .write,
    .read, .write]
  [.plain, .sc, .rel, .acq, .plain, o, o, .acq, l, .rlx, .plain, .rel, .acq, .plain]
  [0, 1, 2, 2, 0, 1, 1, 2, 1, 1, 0, 2, 2, 0]
  [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]
  [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6), (7, 8), (7, 9),
    (7, 10), (7, 11), (8, 9), (8, 10), (8, 11), (9, 10), (9, 11), (10, 11), (12, 13)]
  [(2, 3), (0, 4), (1, 5), (2, 7), (6, 8), (0, 10), (11, 12)]
  [(1, 6), (1, 9), (6, 9), (2, 11), (0, 13)]
  [(5, 6)]

/-- With the orders of the library, the run has no data race. -/
theorem cloneThenMove_ok : (cloneThenMove .acqRel .acq).check = some (true, false) := by
  decide +kernel

/-- When A takes 1 from the count with a relaxed read-modify-write, its clone races with the
write of the next `send`. -/
theorem cloneThenMove_relaxed_rmw : (cloneThenMove .rlx .acq).check = some (true, true) ∧
    (cloneThenMove .rlx .acq).raceOn 4 13 = some true := by decide +kernel

/-- When B reads the count with a relaxed load, the clone of A races with the write of the next
`send`. -/
theorem cloneThenMove_relaxed_load : (cloneThenMove .acqRel .rlx).check = some (true, true) ∧
    (cloneThenMove .acqRel .rlx).raceOn 4 13 = some true := by decide +kernel

/-- Receiver A clones the value and takes 1 from the count with the read-modify-write `a`, `a'`.
Receiver B clones the value, takes the last 1 with the read-modify-write `b`, `b'`, and drops
the value.

Thread 0, the `send`: 0 writes the value, 1 sets the count to 2, 2 sets the number. Thread 1,
receiver A: 3 reads the number, 4 clones the value, 5 and 6 take 1 from the count. Thread 2,
receiver B: 7 reads the number, 8 clones the value, 9 and 10 take 1 from the count and read 1,
11 drops the value. -/
def cloneThenDrop (a a' b b' : Order) : Data 12 := Data.ofLists 12
  [.write, .write, .write, .read, .read, .read, .write, .read, .read, .read, .write, .write]
  [.plain, .sc, .rel, .acq, .plain, a, a', .acq, .plain, b, b', .plain]
  [0, 1, 2, 2, 0, 1, 1, 2, 0, 1, 1, 0]
  [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]
  [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6), (7, 8), (7, 9),
    (7, 10), (7, 11), (8, 9), (8, 10), (8, 11), (9, 10), (9, 11), (10, 11)]
  [(2, 3), (0, 4), (1, 5), (2, 7), (0, 8), (6, 9)]
  [(1, 6), (1, 10), (6, 10), (0, 11)]
  [(5, 6), (9, 10)]

/-- With the orders of the library, the run has no data race. -/
theorem cloneThenDrop_ok :
    (cloneThenDrop .acqRel .acqRel .acqRel .acqRel).check = some (true, false) := by
  decide +kernel

/-- When A takes 1 with a relaxed read-modify-write, its clone races with the drop by B. -/
theorem cloneThenDrop_relaxed_first :
    (cloneThenDrop .rlx .rlx .acqRel .acqRel).check = some (true, true) ∧
    (cloneThenDrop .rlx .rlx .acqRel .acqRel).raceOn 4 11 = some true := by decide +kernel

/-- When the last read-modify-write is a release one with no acquire, the clone of A races with
the drop by B. The drop of a receiver that reads 1 needs the acquire too. -/
theorem cloneThenDrop_release_last :
    (cloneThenDrop .acqRel .acqRel .rlx .rel).check = some (true, true) ∧
    (cloneThenDrop .acqRel .acqRel .rlx .rel).raceOn 4 11 = some true := by decide +kernel

/-- Receiver A clones the value and takes 1 from the count. Then a receiver D is dropped and
takes 1 with the relaxed read-modify-write `d`, `d'`. Then receiver C clones the value, takes
the last 1, and drops the value.

Thread 0, the `send`: 0 writes the value, 1 sets the count to 3, 2 sets the number. Thread 1,
receiver A: 3 reads the number, 4 clones the value, 5 and 6 take 1. Thread 2, the drop of D: 7
and 8 take 1. Thread 3, receiver C: 9 reads the number, 10 clones the value, 11 and 12 take 1
and read 1, 13 drops the value. -/
def cloneDropThenFree (d d' : Order) : Data 14 := Data.ofLists 14
  [.write, .write, .write, .read, .read, .read, .write, .read, .write, .read, .read, .read,
    .write, .write]
  [.plain, .sc, .rel, .acq, .plain, .acqRel, .acqRel, d, d', .acq, .plain, .acqRel, .acqRel,
    .plain]
  [0, 1, 2, 2, 0, 1, 1, 1, 1, 2, 0, 1, 1, 0]
  [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
  [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6), (7, 8), (9, 10),
    (9, 11), (9, 12), (9, 13), (10, 11), (10, 12), (10, 13), (11, 12), (11, 13), (12, 13)]
  [(2, 3), (0, 4), (1, 5), (6, 7), (2, 9), (0, 10), (8, 11)]
  [(1, 6), (1, 8), (1, 12), (6, 8), (6, 12), (8, 12), (0, 13)]
  [(5, 6), (7, 8)]

/-- A relaxed read-modify-write in the drop of D does not break the release sequence from A to
C. So the clone of A happens before the drop of the value by C, and the release part of the
read-modify-write in the drop of a receiver is not needed. -/
theorem cloneDropThenFree_relaxed_drop : (cloneDropThenFree .rlx .rlx).check = some (true, false) := by
  decide +kernel

end Vx
