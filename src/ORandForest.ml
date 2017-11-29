
(******************************************************************************
 * ORandForest
 * A pure OCaml implementation of a random forest classifier based on OC4.5.
 *
 * By Théophile Bastian <contact@tobast.fr>
 * and Noémie Fong (aka. Minithorynque), 2016.
 ******************************************************************************
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************)

module type S = sig
  type c45_data
  type c45_category
  type c45_trainSet
  type randomForest

  val genRandomForest: ?ncores:int -> int -> c45_trainSet -> randomForest
  val classify: randomForest -> c45_data -> c45_category

  val save_to_file: string -> randomForest -> unit
  val restore_from_file: string -> randomForest
end

module Utils = struct
  type filename = string

  let with_out_file (fn: filename) (f: out_channel -> 'a): 'a =
    let output = open_out_bin fn in
    let res = f output in
    close_out output;
    res

  let with_in_file (fn: filename) (f: in_channel -> 'a): 'a =
    let input = open_in_bin fn in
    let res = f input in
    close_in input;
    res

  (* marshal to file *)
  let save (fn: filename) (x: 'a): unit =
    with_out_file fn (fun out ->
        Marshal.to_channel out x [Marshal.No_sharing]
      )

  (* unmarshal from file *)
  let restore (fn: filename): 'a =
    with_in_file fn (fun input ->
        Marshal.from_channel input
      )
end

module Make(X: Oc45.S) = struct
  (***************** DATA TYPES ********************************************)
  type c45_data = X.data
  type c45_category = X.category
  type c45_trainSet = X.trainSet

  module IMap = Map.Make(struct type t=int let compare = compare end)
  type featureMap = int IMap.t
  type randomForest = (X.decisionTree * featureMap) array
  (************* END DATA TYPES ********************************************)
  open X

  let () = Random.self_init ()

  let (<|>) a b =
    (** a|b : generates the list [a ; a+1 ; ... ; b-1] *)
    let rec span b cur =
      if a = b then a::cur
      else span (b-1) (b::cur)
    in span (b-1) []

  let randPick l =
    let card = List.length l in
    let elt = Random.int card in
    List.nth l elt

  let majorityVote (l : int list) =
    (** Returns the most present value in l. If the maximum is not unique,
	returns an arbitrary value among the possible ones. *)
    let counts = List.fold_left
	(fun map x -> IMap.add x
	    ((try IMap.find x map with Not_found -> 0) + 1) map)
	IMap.empty l in
    let cMax,maxarg = IMap.fold (fun arg v (cMax,cArg) ->
	if v > cMax then
	  (v,[arg])
	else if v = cMax then
	  (v,arg::cArg)
	else
	  (cMax,cArg))
	counts (-1,[]) in
    assert (maxarg <> []) ;
    randPick maxarg

  let remapData featMap data =
    let out = Array.make (IMap.cardinal featMap) data.(0) in
    IMap.iter (fun from dest ->
	out.(dest) <- data.(from)) featMap ;
    out

  let classify (forest: randomForest) data =
    let votesList = Array.fold_left (fun cur (tree,ftMap) ->
	(X.classify tree (remapData ftMap data))::cur) [] forest in
    majorityVote votesList

  let genRandomForest ?(ncores = 1) nbTrees (trainset : X.trainSet) : randomForest =
    let trainDataArray = Array.of_list (X.getSet trainset) in
    let randSubsetOf superSize subSize =
      let rec sel selected = function
	| 0 -> selected
	| k ->
	  let el = Random.int superSize in
	  if IMap.mem el selected then
	    sel selected k
	  else
	    sel (IMap.add el (k-1) selected) (k-1)
      in
      sel IMap.empty subSize
    in
    let selectFeatureSubset (trList : X.trainVal list) featCont =
      let subsize = int_of_float (sqrt (float_of_int
				          (X.getNbFeatures trainset))) in
      let selected = randSubsetOf (X.getNbFeatures trainset)
	  subsize in
      (List.fold_left (fun cur x ->
	   { x with data = remapData selected x.data}::cur )
	  [] trList),
      (remapData selected featCont),
      selected
    in
    let generateTree () =
      let nTrainList = List.fold_left (fun cur _ ->
	  let sample = Random.int (Array.length trainDataArray) in
	  (trainDataArray.(sample)) :: cur)
	  [] (0<|> (Array.length trainDataArray)) in
      let trainList, nCont, featMap = selectFeatureSubset nTrainList
	  (X.getFeatContinuity trainset) in

      let nTrainSet = List.fold_left (fun cur x -> X.addData x cur)
	  (X.emptyTrainSet
	     (Array.length ((List.hd trainList).data))
	     (X.getNbCategories trainset)
	     nCont)
	  trainList in
      let ftMaxArray = X.getFeatureMax trainset in
      IMap.iter (fun ft dest ->
          X.setFeatureMax dest ftMaxArray.(ft) nTrainSet) featMap;
      X.c45 nTrainSet, featMap
    in
    if ncores > 1 then
      let units = Array.make nbTrees () in
      Parmap.array_parmap
        ~init:(fun _child_rank -> Random.self_init ())
        ~ncores ~chunksize:1 generateTree units
    else
      Array.init nbTrees (fun i -> generateTree ())

  let save_to_file fn model =
    Utils.save fn model

  let restore_from_file fn =
    Utils.restore fn

end

module IntRandForest = Make(Oc45.IntOc45)
module FloatRandForest = Make(Oc45.FloatOc45)
