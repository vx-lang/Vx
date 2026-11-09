import VxMemoryModel.ScopedCheck

/-!
# The checker as a program

`run_lean.py` gives this program one execution on each line of its input. For each one, the
program prints what `Data.check` or `SData.check` gives, in the form of `#eval`. A line has these
fields, split by `|`. Each field is a list of words split by spaces.

`C n` or `S n`, kinds (`r w f`), orders, places, threads, po, rf, co, rmw, scopes, groups,
barriers. A pair is `a,b` and a group is `subgroup,block,device`. A barrier is `0` or `1`.
`C` uses `Rc11.lean` and needs only the fields up to rmw. `S` uses `Scopes.lean`.
-/

open Vx

deriving instance Inhabited for Kind, Order, Scope

def words (s : String) : List String := (s.splitOn " ").filter (· ≠ "")

def parseNat (s : String) : Except String Nat :=
  match s.toNat? with
  | some v => .ok v
  | none => .error s!"not a number: {s}"

def parseKind : String → Except String Kind
  | "r" => .ok .read
  | "w" => .ok .write
  | "f" => .ok .fence
  | s => .error s!"not a kind: {s}"

def parseOrder : String → Except String Order
  | "plain" => .ok .plain
  | "rlx" => .ok .rlx
  | "acq" => .ok .acq
  | "rel" => .ok .rel
  | "acqRel" => .ok .acqRel
  | "sc" => .ok .sc
  | s => .error s!"not an order: {s}"

def parseScope : String → Except String Scope
  | "subgroup" => .ok .subgroup
  | "block" => .ok .block
  | "device" => .ok .device
  | "system" => .ok .system
  | s => .error s!"not a scope: {s}"

def parsePair (s : String) : Except String (Nat × Nat) :=
  match s.splitOn "," with
  | [a, b] => return (← parseNat a, ← parseNat b)
  | _ => .error s!"not a pair: {s}"

def parseGroup (s : String) : Except String (Nat × Nat × Nat) :=
  match s.splitOn "," with
  | [a, b, c] => return (← parseNat a, ← parseNat b, ← parseNat c)
  | _ => .error s!"not a group: {s}"

def render (r : Option (Bool × Bool)) : String :=
  match r with
  | none => "none"
  | some (a, b) => s!"some ({a}, {b})"

/-- As `Data.ofLists`, but each list becomes an array or a table first. A lookup then takes one
step, not a walk along the list. -/
def fastData (n : Nat) (kind : List Kind) (order : List Order) (loc thread : List Nat)
    (po rf co rmw : List (Nat × Nat)) : Data n :=
  let d := Data.ofLists n kind order loc thread po rf co rmw
  let kind := Array.ofFn d.kind
  let order := Array.ofFn d.order
  let loc := Array.ofFn d.loc
  let thread := Array.ofFn d.thread
  let po := Table.ofFn d.po
  let rf := Table.ofFn d.rf
  let co := Table.ofFn d.co
  let rmw := Table.ofFn d.rmw
  { kind := fun i => kind[i]!, order := fun i => order[i]!, loc := fun i => loc[i]!,
    thread := fun i => thread[i]!, po := po.get, rf := rf.get, co := co.get, rmw := rmw.get }

def fastSData (n : Nat) (kind : List Kind) (order : List Order) (loc thread : List Nat)
    (po rf co rmw : List (Nat × Nat)) (scope : List Scope) (groups : List (Nat × Nat × Nat))
    (barrier : List Bool) : SData n :=
  let d := SData.ofLists n kind order loc thread po rf co rmw scope groups barrier
  let scope := Array.ofFn d.scope
  let barrier := Array.ofFn d.barrier
  { toData := fastData n kind order loc thread po rf co rmw, scope := fun i => scope[i]!,
    group := d.group, barrier := fun i => barrier[i]! }

def checkLine (line : String) : Except String String := do
  let fields := (line.splitOn "|").map words
  let field (i : Nat) : List String := fields.getD i []
  let (mode, n) ← match field 0 with
    | [m, n] => pure (m, ← parseNat n)
    | _ => throw s!"no mode and size: {line}"
  let kind ← (field 1).mapM parseKind
  let order ← (field 2).mapM parseOrder
  let loc ← (field 3).mapM parseNat
  let thread ← (field 4).mapM parseNat
  let po ← (field 5).mapM parsePair
  let rf ← (field 6).mapM parsePair
  let co ← (field 7).mapM parsePair
  let rmw ← (field 8).mapM parsePair
  match mode with
  | "C" => return render (fastData n kind order loc thread po rf co rmw).check
  | "S" =>
    let scope ← (field 9).mapM parseScope
    let groups ← (field 10).mapM parseGroup
    let barrier := (field 11).map (· == "1")
    return render
      (fastSData n kind order loc thread po rf co rmw scope groups barrier).check
  | m => throw s!"not a mode: {m}"

partial def loop (stdin : IO.FS.Stream) (stdout : IO.FS.Stream) : IO UInt32 := do
  let line ← stdin.getLine
  if line.isEmpty then
    return 0
  let line := line.replace "\n" ""
  if line.isEmpty then
    loop stdin stdout
  else
    match checkLine line with
    | .ok r =>
      stdout.putStrLn r
      stdout.flush
      loop stdin stdout
    | .error e =>
      (← IO.getStderr).putStrLn e
      return 1

def main : IO UInt32 := do
  loop (← IO.getStdin) (← IO.getStdout)
