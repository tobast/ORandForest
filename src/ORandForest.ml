
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

(********************* DATA TYPES ********************************************)
module IMap = Map.Make(struct type t=int let compare = compare end)
type featureMap = int IMap.t
type randomForest = (Oc45.decisionTree * featureMap) array
(***************** END DATA TYPES ********************************************)
open Oc45

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
		(Oc45.classify tree (remapData ftMap data))::cur) [] forest in
	majorityVote votesList

let genRandomForest nbTrees (trainset : Oc45.trainSet) : randomForest =
	let trainDataArray = Array.of_list (Oc45.getSet trainset) in
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
	let selectFeatureSubset (trList : Oc45.trainVal list) featCont =
		let subsize = int_of_float (sqrt (float_of_int
			(Oc45.getNbFeatures trainset))) in
		let selected = randSubsetOf (Oc45.getNbFeatures trainset) subsize in
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
		let trainList, nFeat, featMap = selectFeatureSubset nTrainList
			(Oc45.getFeatContinuity trainset) in

		let nTrainSet = List.fold_left (fun cur x -> Oc45.addData x cur)
			(Oc45.emptyTrainSet
				(Array.length ((List.hd trainList).data))
				(Oc45.getNbCategories trainset)
				nFeat)
			trainList in
		Oc45.c45 nTrainSet, featMap
	in

	Array.init nbTrees (fun i -> generateTree ())
	
