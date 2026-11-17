---------------------------- MODULE Rendezvous ----------------------------
(***************************************************************************)
(* How the ranks of a job join for the collectives, in `init` of           *)
(* runtime/vx_collective.h.                                                *)
(*                                                                         *)
(* Rank 0 checks that the id file is not there. It writes the NCCL id to   *)
(* another file and then renames that file to the id file. The other ranks *)
(* wait for the id file and read it, or stop after a time limit. Then each *)
(* rank calls ncclCommInitRank, which returns when every rank called it    *)
(* with the same id.                                                       *)
(*                                                                         *)
(* An id has three values: "new" is the id of this job, "old" is the id of *)
(* an earlier job, and "half" is a file that rank 0 did not finish.        *)
(*                                                                         *)
(* `Bug` names one mistake to put into the protocol. run_tlc.py checks     *)
(* that TLC finds each one. "none" is the protocol of the runtime.         *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
    N,        \* the number of ranks
    OldFile,  \* TRUE: the id file of an earlier job is there at the start
    Faults,   \* TRUE: rank 0 can fail before it writes, and a rank can wait too long
    Bug

Ranks == 0..(N - 1)
Others == Ranks \ {0}
Done == {"joined", "error"}

VARIABLES
    file,     \* the id file: "none", "half", "new" or "old"
    partial,  \* the file that rank 0 writes first: "none", "half" or "new"
    id,       \* the id that each rank holds, or "none"
    pc

vars == <<file, partial, id, pc>>

Init ==
    /\ file = IF OldFile THEN "old" ELSE "none"
    /\ partial = "none"
    /\ id = [r \in Ranks |-> "none"]
    /\ pc = [r \in Ranks |-> IF r = 0 THEN "check" ELSE "poll"]

-----------------------------------------------------------------------------
(* Rank 0.                                                                 *)

InPlace == Bug = "write_the_id_in_place"

\* An id file that is there may hold the id of an earlier job.
Check ==
    /\ pc[0] = "check"
    /\ pc' = [pc EXCEPT ![0] = IF file # "none" /\ Bug # "no_check_for_an_old_file"
                               THEN "error" ELSE "write"]
    /\ UNCHANGED <<file, partial, id>>

\* fwrite is not one step, so a reader can see half an id.
WriteHalf ==
    /\ pc[0] = "write"
    /\ IF InPlace THEN file' = "half" /\ UNCHANGED partial
                  ELSE partial' = "half" /\ UNCHANGED file
    /\ pc' = [pc EXCEPT ![0] = "finish"]
    /\ UNCHANGED id

WriteRest ==
    /\ pc[0] = "finish"
    /\ IF InPlace THEN file' = "new" /\ UNCHANGED partial
                  ELSE partial' = "new" /\ UNCHANGED file
    /\ id' = [id EXCEPT ![0] = "new"]
    /\ pc' = [pc EXCEPT ![0] = IF InPlace THEN "init" ELSE "rename"]

Rename ==
    /\ pc[0] = "rename"
    /\ file' = IF Bug = "rank_0_removes_the_file_before_its_init" THEN "none" ELSE partial
    /\ partial' = "none"
    /\ pc' = [pc EXCEPT ![0] = "init"]
    /\ UNCHANGED id

\* ncclGetUniqueId fails, or rank 0 cannot write the file.
Fail ==
    /\ Faults
    /\ pc[0] \in {"check", "write", "finish"}
    /\ pc' = [pc EXCEPT ![0] = "error"]
    /\ UNCHANGED <<file, partial, id>>

-----------------------------------------------------------------------------
(* The other ranks.                                                        *)

Read(r) ==
    /\ pc[r] = "poll"
    /\ file # "none"
    /\ id' = [id EXCEPT ![r] = file]
    /\ pc' = [pc EXCEPT ![r] = "init"]
    /\ UNCHANGED <<file, partial>>

\* The time limit of `read_unique_id`. With no faults, rank 0 is never that slow.
GiveUp(r) ==
    /\ pc[r] = "poll"
    /\ file = "none"
    /\ Faults
    /\ Bug # "a_rank_waits_with_no_time_limit"
    /\ pc' = [pc EXCEPT ![r] = "error"]
    /\ UNCHANGED <<file, partial, id>>

-----------------------------------------------------------------------------
(* ncclCommInitRank.                                                       *)

Join ==
    /\ \A r \in Ranks : pc[r] = "init" /\ id[r] = id[0]
    /\ pc' = [r \in Ranks |-> "joined"]
    /\ UNCHANGED <<file, partial, id>>

\* The model assumes that NCCL stops with an error when its ranks can never meet.
NeverMeets(r) == \E q \in Ranks : pc[q] = "error" \/ (pc[q] = "init" /\ id[q] # id[r])

InitFails(r) ==
    /\ pc[r] = "init"
    /\ NeverMeets(r)
    /\ pc' = [pc EXCEPT ![r] = "error"]
    /\ UNCHANGED <<file, partial, id>>

-----------------------------------------------------------------------------

\* A rank that waits forever is a state with no step. EachRankEnds reports it.
Next ==
    \/ Check \/ WriteHalf \/ WriteRest \/ Rename \/ Fail \/ Join
    \/ \E r \in Others : Read(r) \/ GiveUp(r)
    \/ \E r \in Ranks : InitFails(r)
    \/ UNCHANGED vars

Spec ==
    /\ Init /\ [][Next]_vars
    /\ WF_vars(Check) /\ WF_vars(WriteHalf) /\ WF_vars(WriteRest) /\ WF_vars(Rename)
    /\ WF_vars(Join)
    /\ \A r \in Others : WF_vars(Read(r)) /\ WF_vars(GiveUp(r))
    /\ \A r \in Ranks : WF_vars(InitFails(r))

-----------------------------------------------------------------------------
(* The claims.                                                             *)

TypeOK ==
    /\ file \in {"none", "half", "new", "old"}
    /\ partial \in {"none", "half", "new"}
    /\ id \in [Ranks -> {"none", "half", "new", "old"}]

\* No rank reads half an id.
NoHalfId == \A r \in Ranks : id[r] # "half"

\* A rank can read the id of an earlier job. Then rank 0 must stop at its check.
AnOldIdStopsRankZero == (\E r \in Ranks : id[r] = "old") => pc[0] \notin {"init", "joined"}

\* Each rank joins or stops with an error.
EachRankEnds == <>(\A r \in Ranks : pc[r] \in Done)

\* With no old file and no fault, every rank joins.
EveryRankJoins == <>((\A r \in Ranks : pc[r] = "joined") \/ OldFile \/ Faults)

=============================================================================
