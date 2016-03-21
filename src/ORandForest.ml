
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
type randomForest = Oc45.decisionTree array
(***************** END DATA TYPES ********************************************)
open Oc45

let () = Random.self_init ()

let (<|>) a b =
	(** a|b : generates the list [a ; a+1 ; ... ; b-1] *)
	let rec span b cur =
		if a = b then a::cur
			else span (b-1) (b::cur)
	in span (b-1) []


module IMap = Map.Make(struct type t=int let compare = compare end)
let majorityVote (l : int list) =
	(** Returns the most present value in l. If the maximum is not unique,
		returns an arbitrary value among the possible ones. *)
	let counts = List.fold_left
		(fun map x -> IMap.add x
			((try IMap.find x map with Not_found -> 0) + 1) map)
		IMap.empty l in
	let _,maxarg = IMap.fold (fun arg v (cMax,cArg) ->
		if v > cMax then (v,arg) else (cMax,cArg)) counts (-1,-1) in
	maxarg

let classify forest data =
	let votesList = List.fold_left (fun cur x ->
		(Oc45.classify x data)::cur) [] forest in
	majorityVote votesList

let genRandomForest nbTrees (trainset : Oc45.trainSet) =
	let trainDataArray = Array.of_list (Oc45.getSet trainset) in
	let randSubsetOf superSize subSize =
		let selected = Array.make superSize false in
		let rec elim = function
		| 0 -> ()
		| k ->
			let el = Random.int superSize in
			if selected.(el) then
				elim k
			else begin
				selected.(el) <- true;
				elim (k-1)
			end
		in
		elim (superSize-subSize) ;
		selected
	in
	let arraySubset arr selector nbSel =
		let out = Array.make nbSel arr.(0) in
		let cPos = ref 0 in
		Array.iteri (fun i v -> (match selector.(i) with
			| true ->
				out.(!cPos) <- v ;
				cPos := !cPos + 1
			| false -> ())) arr;
		out
	in
	let selectFeatureSubset (trList : Oc45.trainVal list) =
		let subsize = int_of_float (sqrt (float_of_int
			(Oc45.getNbFeatures trainset))) in
		let selected = randSubsetOf (Oc45.getNbFeatures trainset) subsize in
		List.fold_left (fun cur x ->
			{ x with data = arraySubset x.data selected subsize}::cur )
			[] trList
	in
	let generateTree () =
		let nTrainList = List.fold_left (fun cur _ ->
				let sample = Random.int (Array.length trainDataArray) in
				(trainDataArray.(sample)) :: cur)
			[] (0<|> (Array.length trainDataArray)) in
		let trainList = selectFeatureSubset nTrainList in

		List.fold_left (fun cur x -> Oc45.addData x cur)
			(Oc45.emptyTrainSet
				(Array.length ((List.hd trainList).data))
				(Oc45.getNbCategories trainset)
				(Oc45.getFeatContinuity trainset))
			trainList
	in

	Array.init nbTrees (fun _ -> generateTree ())
	
