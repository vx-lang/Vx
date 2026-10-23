import VxMemoryModel.Check
import VxMemoryModel.Counter

/-!
# Two `fetchAdd`s that read the same value

Thread 0 sets a counter to 0 (event 0). Threads 1 and 2 each add 1 with a `fetchAdd`: a read
(events 1 and 3) and a write (events 2 and 4). Both reads read from event 0, so both threads get
the number 0.

`Consistent` allows this when `co` does not order the two writes of 1. RC11 orders them, and then
the execution is not consistent. So `Counter.tickets` needs `CoOrder`.
-/

namespace Vx

def twoTickets (co : List (Nat × Nat)) : Data 5 := Data.ofLists 5
  [.write, .read, .write, .read, .write] [.sc, .sc, .sc, .sc, .sc] [0, 0, 0, 0, 0]
  [0, 1, 1, 2, 2] [(1, 2), (3, 4)] [(0, 1), (0, 3)] co [(1, 2), (3, 4)]

-- `co` does not order events 2 and 4: consistent, with no data race.
#guard (twoTickets [(0, 2), (0, 4)]).check == some (true, false)

-- `co` orders event 2 before event 4: not consistent.
#guard (twoTickets [(0, 2), (0, 4), (2, 4)]).check == some (false, false)

end Vx
