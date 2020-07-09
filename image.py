from PIL import Image
import yaml
import numpy as np
from typing import Tuple, Sequence, Dict, Set
import itertools as it
import networkx as nx
from operator import itemgetter, attrgetter
from collections import defaultdict, namedtuple
from functools import reduce
import matplotlib.pyplot as plt

"""
You will need: PIL, numpy, networkx, matplotlib, pygraphviz
Pygraphviz requires libgraphviz-dev package on ubuntu

Bibliography

Planning Algorithms online textbook
http://planning.cs.uiuc.edu/node260.html
 6.2 Polygonal Obstacle Regions 
    6.2.1 Representation
    6.2.2 Vertical Cell Decomposition
        Defining the vertical decomposition
        General position issues
        Defining the roadmap
        Solving a query
        Computing the decomposition
        Plane-sweep principle
        Algorithm execution 
    6.2.3 Maximum-Clearance Roadmaps
    6.2.4 Shortest-Path Roadmaps 
http://planning.cs.uiuc.edu/node352.html
 Boustrophedon decomposition 

Coverage of Known Spaces: The Boustrophedon Cellular Decomposition (use library acct; Choset)
https://doi.org/10.1023/A:1008958800904 

Efficient complete coverage of a known arbitrary environment with applications to aerial operations (use lbrary acct; Xu, Viriyasuthee & Rekleitis)
https://link.springer.com/article/10.1007/s10514-013-9364-x

Sensor-based Coverage of Unknown Environments: Incremental Construction of Morse Decompositions (use library acct; Acar & Choset)
https://doi.org/10.1177%2F027836402320556368

*** Coverage Path Planning: The Boustrophedon Cellular Decomposition (Choset & Pignon)
https://pdfs.semanticscholar.org/8df2/e8ed0410c3ae5b0f56c4367b740e91c23d73.pdf

*** Optimal Coverage of a Known Arbitrary Environment (Mannadiar & Rekleitis)
http://msdl.cs.mcgill.ca/people/raphael/files/yiannis_coverage.pdf

Robotic Motion Planning: Cell Decompositions (lecture slides; Choset)
https://www.cs.cmu.edu/~motionplanning/lecture/Chap6-CellDecomp_howie.pdf

Trapezoidal decomposition (author unk)
https://www2.cs.duke.edu/courses/spring02/cps237/DKnotes/n19.pdf

"""

# Some useful type annotations
BoundaryList = Sequence[Tuple[int]]
AdjacencyList = Dict[int, Set[int]]

SliceInfo = namedtuple("SliceInfo", (
    "connectivity", "x_left", "x_right", "y_groups"
))

def load_image(name: str) -> Tuple[np.ndarray, Dict]:
    """
    Load and return image bitmap and yaml.
    """
    try: 
        with open(name + ".yaml") as f:
            config = yaml.safe_load(f)
    except:
        print("\x1b[41mCannot open yaml\x1b[0m")
        raise
    
    try:
        image = Image.open(name + ".pgm")
    except:
        print("\x1b[41mCannot open pgm\x1b[0m")
        raise
    
    arr = np.array(image.getdata(), dtype="uint8")
    arr.shape = image.size
    # image is transposed so that a[X,Y] indexing works as expected, but
    # more importantly, so that it is stored in column major order so when
    # we iterate over the array, we get columns and not rows
    arr = arr.T

    return arr, config


def slice_count(slic: np.ndarray, threshold: int) -> Tuple[int, BoundaryList]:
    """
    Count the segments in a slice of the bitmap. A pixel with value >= the 
    threshold is considered to be unobstructed. Returns the connectivity and
    a tuple of start and end pairs of each segment, exclusive of the obstacle 
    parts of the slice.

    So, if the slice looks like this:
    
     0 1 2 3 4 5 6 7 8 9
    +-+-+-+-+-+-+-+-+-+-+
    | |x|x|x|x| | |x|x|x|
    +-+-+-+-+-+-+-+-+-+-+

    the return value is:
    (2, ((1,4), (7,9)))
    """
    slice_begin = []
    slice_end = []
    for i in range(len(slic)-1):
        if slic[i+1] >= threshold and slic[i] < threshold:
            # begin slice (append the coord of the open pixel)
            slice_begin.append(i+1)
        
        if slic[i+1] < threshold and slic[i] >= threshold:
            # end slice (again, open pixel)
            slice_end.append(i)
    
    # reconcile slice begin and end
    if len(slice_begin) != len(slice_end):
        raise ValueError("Unterminated slices")

    return len(slice_begin), tuple(zip(slice_begin, slice_end))


def sweep(arr: np.ndarray, threshold: int) \
        -> Tuple[Sequence[int], Sequence[BoundaryList], Sequence[int]]:
    """
    Sweep the bitmap to find out connectivity and segment boundaries for each
    slice. This is basically slice_count but for each slice in a 2d array.
    Yields: connectivity, boundaries, x coordinates of each slice
    """
    for i, slic in enumerate(arr):
        connectivity, slice_bound = slice_count(slic, threshold)
        yield connectivity, slice_bound, i


def do_segments_overlap(left, right):
    if left[0] <= right[1] <= left[1]:
        return True
    elif left[0] <= right[0] <= left[1]:
        return True
    elif right[0] <= left[1] <= right[1]:
        return True
    elif right[0] <= left[0] <= right[1]:
        # probably not needed
        return True
    else:
        return False


def unzip(a):
    if not isinstance(a, (list, tuple)):
        a = tuple(a)
    return list(zip(*a))


def take_column(a, i):
    out = []
    for elem in a:
        out.append(elem[i])
    return out


def find_adjacents(left: BoundaryList, right: BoundaryList) -> AdjacencyList:
    """
    Given left and right lists of segments, determine the adjacency relations
    between segments on the left and right. 

    Input: 
        left: ((0, 10), (20, 30), (33, 36))
        right: ((0, 10), (20, 23), (27, 29), (40, 50))
    Output: {0: {0}, 1: {1, 2}, 3: {}, None: {3}}

    Note that keys in the output use the indices from the left list, and the 
    values use the indices from the right. The None key represents no 
    connecting segment on the left, and the empty set represents no connecting
    segment on the right.
    """
    # Handle special cases first
    if len(left) == 0 and len(right) == 0:
        return {}
    
    if len(left) == 0:
        # right is not [], so all right segments are gained
        return {None: tuple(range(len(right)))}
    
    if len(right) == 0:
        # left is not [], so all left segments are lost
        return {x: tuple() for x in range(len(left))}
    
    # Start at the topmost segments for left and right
    li = 0
    ri = 0

    out = defaultdict(set)

    # This is set to false when we reach the bottommost segments on the left 
    # and right
    lrun = True
    rrun = True

    while lrun and rrun:
        if do_segments_overlap(left[li], right[ri]):
            # overlap means right segment is reachable from left
            out[li].add(ri)

        # Which segment ends first?
        if left[li][1] < right[ri][1]:
            # Current left segment ends first, advance left if possible
            if li + 1 < len(left):
                li += 1
            else:
                # Done for the left
                lrun = False
        else:
            # Right segment ends first (or, they end at the same time)
            if ri + 1 < len(right):
                ri += 1
            else:
                # Done for the right
                rrun = False
    
    # Check for unconnected segments on the left and add them back
    for lj in range(len(left)):
        if lj not in out:
            out[lj] = set()
    
    # ...And on the right
    right_union = reduce(lambda x, y: x.union(y), out.values())
    for rj in range(len(right)):
        if rj not in right_union:
            out[None].add(rj)

    return out


def sweep_for_slices(arr: np.ndarray, start: Sequence[int], threshold: int) \
        -> Tuple[Sequence[SliceInfo], Sequence[AdjacencyList]]:
    """
    Sweep the bitmap but also record down slices and adjacency lists.
    """
    # list of SliceInfo objects
    slices = []
    # The slice adjacency lists resulting from the above `slices` list
    sweep_adjacency_lists = []

    # groupby collapses continuous runs of items into a single element.
    # >>> [(k, ''.join(g)) for k, g in it.groupby("AAAABBCCCBDD")]
    # [('A', "AAAA"), ('B', "BB"), ('C', "CCC"), ('B', "B"), ('D', "DD")]
    # The key accesses the connectivity in each item returned by the
    # enumerate(sweep(arr, threshold)) iterator.
    # sweep() gives us connectivity and y-segment boundaries.
    # enumerate gives us the x-coordinate of each slice.
    for k, g in it.groupby(sweep(arr, threshold), key=itemgetter(0)):
        # force iterator to list
        g = list(g)
        sliceinfo = SliceInfo(
            connectivity=k,
            x_left=g[0][2],
            x_right=g[-1][2],
            y_groups=[slic[1] for slic in g]
        )
        slices.append(sliceinfo)

        if len(slices) == 1:
            # no point in trying to find adjacents
            continue

        end_left = slices[-2].y_groups[-1]
        start_right = sliceinfo.y_groups[0]
        adjacency = find_adjacents(end_left, start_right)
        sweep_adjacency_lists.append(adjacency)

    return slices, sweep_adjacency_lists


def find_events_from_adjlist(adj: AdjacencyList) -> Dict[str, AdjacencyList]:
    """
    Given an AdjacencyList from find_adjacents, determine the "events" occuring
    at the transition from left to right.

    There are five kinds of events:
    - "Continue" a segment to another. This is the degenerate case, nothing 
      interesting happens in this case and the overall Reeb/adjacency graph 
      doesn't need to be updated. However, the cell boundaries need to be 
      updated whenever we have a continue event (see Choset & Pignon). 
    
    The other four will result in updating the Reeb/adjacency graph:
    - "Split" of a segment into two
    - "Merge" of two segments into one
    - "Loss" of a segment (as in a concave shape)
    - "Gain" of a segment (as in a concave shape)

    Note that the events correspond to critical points. Note also that the 
    split and merge events correspond to points in contact with 3 cells but 
    loss and gain events correspond to points in contact with 1 cell. (See 
    Mannadiar & Rekleitis)
    """
    # reverse the adjacency list. Losses will be, uh, lost, but we just want 
    # to find out the merge events from this
    rev = defaultdict(set)
    for k, v_set in adj.items():
        for v in v_set:
            rev[v].add(k)
    
    # The events can be recognised from certain patterns in the adjacency 
    # relationship
    split = {k: v_set for k, v_set in adj.items() if len(v_set) > 1}
    merge = {k: v_set for k, v_set in rev.items() if len(v_set) > 1}
    loss = {k: v_set for k, v_set in adj.items() if len(v_set) == 0}
    gain = {k: v_set for k, v_set in adj.items() if k is None}

    out = {
        "split": split,
        "merge": merge,
        "loss": loss,
        "gain": gain
    }

    # Remove events that didn't happen
    return {k: v for k, v in out.items() if len(v) > 0}


def build_cell_graph(slices: Sequence[SliceInfo],
        adjs: Sequence[AdjacencyList]) -> nx.Graph:
    """
    Convert a sequence of cell adjacency lists to a graph. Cells now have 
    boundaries.

    Since changes to the adjacency graph only occur at changes of connectivity 
    as we sweep the bitmap, we should be able to construct the adjacency graph
    given a sequence of cell adjacency lists, which describe what is happening at the places where connectivity changes. 

    However, the cell adjacency lists won't tell us where the cells are in the 
    bitmap, so we need the slice info list as well to reconstruct this.

    Each node in the returned graph has attributes:
    - x_left: Inclusive leftmost x coordinate of cell
    - x_right: Inclusive rightmost x coordinate of cell
    - y_list: List of tuple of (upper, lower) y-coordinates for each 
              x-coordinate.
    
    Postconditions for each node:
    - len(y_list) == x_right - x_left + 1
    - y_list describes a shape with a single continuous area
    """
    graph = nx.DiGraph()
    # the frontier maps numbers in the event adjacency list to actual
    # node numbers in the graph
    frontier = []
    # Next available node number
    i = 0

    assert slices[0].connectivity == 0 and slices[-1].connectivity == 0, \
        "First and last slices should have zero connectivity"
    
    # Remove the slices with zero connectivity
    slices = slices[1:-1]
    
    assert len(slices) > 0, "Something is really wrong"

    for adj, slic in zip(adjs, slices):
        # These events are how we update the Reeb graph. An event means a 
        # change in connectivity that will result in cells changing.
        events = find_events_from_adjlist(adj)
        modified = []
        for event_name, event_adj in events.items():
            assert len(event_adj) == 1, "too many conn. changes"

            if event_name == "split":
                [[left, right]] = event_adj.items()
                # Get the node number of the left cell.
                # This cell is going to split into two, so it will be replaced
                # in the frontier by its descendant cells.
                pred = frontier.pop(left)
                # modified = list(right)
                for succ in right:
                    # Add new nodes on the right 
                    graph.add_node(i, x_left=slic.x_left, x_right=slic.x_right,
                                   y_list=take_column(slic.y_groups, succ))
                    # Add edges from left to right
                    graph.add_edge(pred, i)
                    # update the frontier to include the descendant cells and 
                    # their node numbers
                    frontier.insert(succ, i)
                    # Mark this node as modified, so we don't extend it again 
                    # later
                    modified.append(i)
                    i += 1

            elif event_name == "merge":
                # The order is inverted for the adjacency relation
                # for easier representation
                [[right, left]] = event_adj.items()
                # modified = list(right)
                # need to delete from the frontier in reverse order,
                # so that we don't disturb the other elements as we delete 
                # the ones in front
                preds = [frontier.pop(pred)
                         for pred in sorted(left, reverse=True)]

                # Create the new node
                graph.add_node(i, x_left=slic.x_left, x_right=slic.x_right,
                               y_list=take_column(slic.y_groups, right))

                # Add the edges from left to right
                for pred in preds:
                    graph.add_edge(pred, i)
                # Only one new node is created, and only one descendant cell
                # is inserted into the frontier
                frontier.insert(right, i)
                modified.append(i)
                i += 1

            elif event_name == "loss":
                [[left, _]] = event_adj.items()
                # Don't change modified

                # No changes to the graph needed, but this node is removed
                # permanently from the frontier
                frontier.pop(left)

            elif event_name == "gain":
                [[_, right]] = event_adj.items()
                # No edges added to graph yet
                for succ in right:
                    frontier.insert(succ, i)
                    modified.append(i)
                    graph.add_node(i, x_left=slic.x_left, x_right=slic.x_right,
                                   y_list=take_column(slic.y_groups, succ))
                    i += 1

            else:
                raise NotImplementedError()
        # Finally, extend all the cells touching the frontier that weren't 
        # modified
        for succ in frontier:
            if succ not in modified:
                graph.nodes[succ]["x_right"] = slic.x_right
                graph.nodes[succ]["y_list"].extend(
                    take_column(slic.y_groups, frontier.index(succ)))
    
    # Postcondition
    for _, attrs in graph.nodes(data=True):
        assert len(attrs["y_list"]) == attrs["x_right"] - attrs["x_left"] + 1
        continuity = []
        for i in range(len(attrs["y_list"]) - 1):
            continuity.append(
                do_segments_overlap(attrs["y_list"][i], attrs["y_list"][i+1]))
        assert all(continuity)

    return graph


def relabel_inplace_fixed(G, mapping):
    """
    Same as networkx.relabel._relabel_inplace, but fixed to avoid overwriting 
    edges if two edges result in the same tails and heads
    See https://github.com/networkx/networkx/issues/4058
    """
    old_labels = set(mapping.keys())
    new_labels = set(mapping.values())
    if len(old_labels & new_labels) > 0:
        # labels sets overlap
        # can we topological sort and still do the relabeling?
        D = nx.DiGraph(list(mapping.items()))
        D.remove_edges_from(nx.selfloop_edges(D))
        try:
            nodes = reversed(list(nx.topological_sort(D)))
        except nx.NetworkXUnfeasible:
            raise nx.NetworkXUnfeasible('The node label sets are overlapping '
                                        'and no ordering can resolve the '
                                        'mapping. Use copy=True.')
    else:
        # non-overlapping label sets
        nodes = old_labels

    multigraph = G.is_multigraph()
    directed = G.is_directed()

    for old in nodes:
        try:
            new = mapping[old]
        except KeyError:
            continue
        if new == old:
            continue
        try:
            G.add_node(new, **G.nodes[old])
        except KeyError:
            raise KeyError("Node %s is not in the graph" % old)
        if multigraph:
            new_edges = [(new, new if old == target else target, key, data)
                         for (_, target, key, data)
                         in G.edges(old, data=True, keys=True)]
            if directed:
                new_edges += [(new if old == source else source, new, key, data)
                              for (source, _, key, data)
                              in G.in_edges(old, data=True, keys=True)]
            # New code
            for i, (tail, head, key, data) in enumerate(new_edges):
                if head in G[tail] and key in G[tail][head]:
                    next_key = max(k for k in G[tail][head]) + 1
                    new_edges[i] = (tail, head, next_key, data)
            # End new code

        else:
            new_edges = [(new, new if old == target else target, data)
                         for (_, target, data) in G.edges(old, data=True)]
            if directed:
                new_edges += [(new if old == source else source, new, data)
                              for (source, _, data) in G.in_edges(old, data=True)]

        G.remove_node(old)
        G.add_edges_from(new_edges)
    return G


def build_reeb_graph(adj_gr: nx.Graph, adjs: Sequence[AdjacencyList]):
    """
    Construct the Reeb graph from the adjacency lists.
    """
    # Newly-created object()s are inserted into the graph as dummy nodes. Those
    # dummy objects are then inserted into the frontier. That way, we don't 
    # have to make a decision on which critical point index to assign them, or 
    # even what kind of critical point should be assigned.
    # (A newly-created object() will only ever compare equal to itself.)

    # Why MultiDiGraph? In the adjacency graph, there can't be more than one 
    # edge between nodes. Equivalently, there can't be more than one way to 
    # reach a neighbour of any cell. (The boustrophedon decomposition 
    # guarantees that, since cells don't overlap.) However, there isn't such a
    # guarantee for the Reeb graph, and indeed there can be more than one path
    # to reach two critical points next to each other.
    reeb_gr = nx.MultiDiGraph()
    frontier = []

    # next available critical point index
    node_i = 0

    # next available cell index
    edge_i = 0

    # The order of assigning free indices to edges is exactly the same as the
    # order of assigning to nodes in the adjacency graph. So the edges in the 
    # Reeb graph have correspondence with the cells.
    for adj in adjs:
        events = find_events_from_adjlist(adj)
        for event_name, event_adj in events.items():
            # assert len(event_adj) == 1, "too many conn. changes"

            if event_name == "split":
                [[left, right]] = event_adj.items()
                
                pred = frontier.pop(left)
                # assert type(pred) == object
                # Replace the dummy object with the next available node number
                nx.relabel_nodes(reeb_gr, {pred: node_i}, copy=False)
                
                for succ in right:
                    obj = object()
                    reeb_gr.add_edge(node_i, obj, cell=edge_i)
                    frontier.insert(succ, obj)

                    edge_i += 1
                
                node_i += 1

            elif event_name == "merge":
                # The order is inverted for the adjacency relation
                # for easier representation
                [[right, left]] = event_adj.items()

                # need to delete from the frontier in reverse order,
                # so that we don't disturb the other elements as we delete 
                # the ones in front
                preds = [frontier.pop(pred)
                         for pred in sorted(left, reverse=True)]
                # order is backward now...
                preds.reverse()

                relabel_inplace_fixed(reeb_gr, {pred: node_i for pred in preds})

                # Only one new node is created, and only one descendant cell
                # is inserted into the frontier
                obj = object()
                reeb_gr.add_edge(node_i, obj, cell=edge_i)
                frontier.insert(right, obj)

                node_i += 1
                edge_i += 1

            elif event_name == "gain":
                [[_, right]] = event_adj.items()

                reeb_gr.add_node(node_i)
                for succ in sorted(right):
                    obj = object()
                    reeb_gr.add_edge(node_i, obj, cell=edge_i)
                    frontier.insert(succ, obj)

                    edge_i += 1

                node_i += 1

            elif event_name == "loss":
                [[left, _]] = event_adj.items()

                obj = frontier.pop(left)
                nx.relabel_nodes(reeb_gr, {obj: node_i}, copy=False)
                node_i += 1

            else:
                raise NotImplementedError()

    # Postcondition (FIXME: not actually met now)
    for cell in adj_gr.nodes():
        assert any(cell == attrs["cell"]
                   for _, _, attrs in reeb_gr.edges(data=True)), \
            f"Cell {cell} not in Reeb graph!"

    return reeb_gr


def graph_labels(gr: nx.Graph) -> Dict[int, str]:
    return {node: f'<{node}>\nx: {attrs["x_left"]}-{attrs["x_right"]}\n'
                  f'y: {attrs["y_list"][0]}...{attrs["y_list"][-1]}'
            for node, attrs in gr.nodes(data=True)}


if __name__ == "__main__":
    arr, config = load_image("test")
    slices, adjs = sweep_for_slices(arr, (0,0), 250)
    graph = build_cell_graph(slices, adjs)
    reeb = build_reeb_graph(graph, adjs)
    # pos = nx.drawing.nx_agraph.pygraphviz_layout(graph, prog="dot")
    pos = nx.drawing.nx_agraph.pygraphviz_layout(reeb, prog="neato")
    # labels = graph_labels(graph)
    # nx.draw(graph, with_labels=True, cmap=plt.cm.Paired, node_color=range(12), 
    #         node_size=800, pos=pos, labels=labels)

    nx.draw(reeb, with_labels=True, cmap=plt.cm.Paired, node_color=range(10),
            node_size=800, pos=pos)
    edge_labels = {(u,v): attrs["cell"] for u,v,attrs in reeb.edges(data=True)}
    nx.draw_networkx_edge_labels(reeb, pos=pos, font_color="red",
                                 edge_labels=edge_labels)
    plt.show()
    print("Break here")