# encoding: utf-8
from __future__ import division
from __future__ import print_function

"Debug mode on/off"
# from pudb import set_trace; set_trace()

import string
import random
import psutil
import pickle
import os

from random import choice


from operator import itemgetter

import networkx as nx

__all__ = [
    'maximum_spanning_arborescence', 'minimum_spanning_arborescence',
    'Edmonds'
]

KINDS = set(['max', 'min'])

STYLES = {
    'arborescence': 'arborescence',
    'spanning arborescence': 'arborescence'
}

INF = float('inf')

def random_string(L=15, seed=None):
    random.seed(seed)
    return ''.join([random.choice(string.ascii_letters) for n in range(L)])

def _min_weight(weight):
    return -weight

def _max_weight(weight):
    return weight

class MultiDiGraph_EdgeKey(nx.MultiDiGraph):
    """
    MultiDiGraph which assigns unique keys to every edge.

    Adds a dictionary edge_index which maps edge keys to (u, v, data) tuples.

    This is not a complete implementation. For Edmonds algorithm, we only use
    add_node and add_edge, so that is all that is implemented here. During
    additions, any specified keys are ignored---this means that you also
    cannot update edge attributes through add_node and add_edge.

    Why do we need this? Edmonds algorithm requires that we track edges, even
    as we change the head and tail of an edge, and even changing the weight
    of edges. We must reliably track edges across graph mutations.

    """
    def __init__(self, data=None, **attr):
        cls = super(MultiDiGraph_EdgeKey, self)
        cls.__init__(data=data, **attr)

        self._cls = cls
        self.edge_index = {}

    def remove_node(self, n):
        keys = set([])
        for keydict in self.pred[n].values():
            keys.update(keydict)
        for keydict in self.succ[n].values():
            keys.update(keydict)

        for key in keys:
            del self.edge_index[key]

        self._cls.remove_node(n)

    def remove_nodes_from(self, nbunch):
        for n in nbunch:
            self.remove_node(n)

    def add_edge(self, u, v, key, attr_dict=None, **attr):
        """
        Key is now required.

        """
        if key in self.edge_index:
            uu, vv, _ = self.edge_index[key]
            if (u != uu) or (v != vv):
                raise Exception("Key {0!r} is already in use.".format(key))

        self._cls.add_edge(u, v, key=key, attr_dict=attr_dict, **attr)
        self.edge_index[key] = (u, v, self.succ[u][v][key])

    def add_edges_from(self, ebunch, attr_dict=None, **attr):
        raise NotImplementedError

    def remove_edge_with_key(self, key):
        try:
            u, v, _  = self.edge_index[key]
        except KeyError:
            raise KeyError('Invalid edge key {0!r}'.format(key))
        else:
            del self.edge_index[key]
            self._cls.remove_edge(u, v, key)

    def remove_edges_from(self, ebunch):
        raise NotImplementedError

def get_path(G, u, v):
    """
    Returns the edge keys of the unique path between u and v.

    This is not a generic function. G must be a branching and an instance of
    MultiDiGraph_EdgeKey.

    """
    nodes = nx.shortest_path(G, u, v)
    # We are guaranteed that there is only one edge connected every node
    # in the shortest path.

    def first_key(i, vv):
        # Needed for 2.x/3.x compatibilitity
        keys = G[nodes[i]][vv].keys()
        # Normalize behavior
        keys = list(keys)
        return keys[0]

    edges = [first_key(i, vv) for i, vv in enumerate(nodes[1:])]
    return nodes, edges

class Edmonds(object):
    """
    Edmonds algorithm for spanning arborescences.

    """
    def __init__(self, G, seed=None):
        self.G_original = G
        self.G_original_class = self.G_original.__class__()
  
        # The final answer.
        self.edges = []

        # Since we will be creating graphs with new nodes, we need to make
        # sure that our node names do not conflict with the real node names.
        self.template = random_string(seed=seed) + '_{0}'

    def _init(self, attr, default, kind, style):
        if kind not in KINDS:
            raise nx.NetworkXException("Unknown value for `kind`.")

        # Store inputs.
        self.attr = attr
        self.default = default
        self.kind = kind
        self.style = style

        # Determine how we are going to transform the weights.
        if kind == 'min':
            self.trans = trans = _min_weight
        else:
            self.trans = trans = _max_weight

        if attr is None:
            # Generate a random attr the graph probably won't have.
            attr = random_string()

        # This is the actual attribute used by the algorithm.
        self._attr = attr

        # The object we manipulate at each step is a multidigraph.
        self.G = G = MultiDiGraph_EdgeKey()
        for key, (u, v, data) in enumerate(self.G_original.edges(data=True)):
            d = {attr: trans(data.get(attr, default))}
            G.add_edge(u, v, key, **d)

        self.G_original = None
        self.level = 0

        # These are the "buckets" from the paper.
        #
        # As in the paper, G^i are modified versions of the original graph.
        # D^i and E^i are nodes and edges of the maximal edges that are
        # consistent with G^i. These are dashed edges in figures A-F of the
        # paper. In this implementation, we store D^i and E^i together as a
        # graph B^i. So we will have strictly more B^i than the paper does.
        self.B = MultiDiGraph_EdgeKey()
        self.B.edge_index = {}
        self.uf = nx.utils.UnionFind()

        # A list of lists of edge indexes. Each list is a circuit for graph G^i.
        # Note the edge list will not, in general, be a circuit in graph G^0.
        # Stores the index of the minimum edge in the circuit found in G^i and B^i.
        # The ordering of the edges seems to preserve the weight ordering from G^0.
        # So even if the circuit does not form a circuit in G^0, it is still true
        # that the minimum edge of the circuit in G^i is still the minimum edge
        # in circuit G^0 (depsite their weights being different).
        self.minedge_circuit = []


    def find_optimum(self, attr='weight', default=1, kind='max', style='branching'):
        """
        Parameters
        ----------
        attr : str
            The edge attribute used to in determining optimality.
        default : float
            The value of the edge attribute used if an edge does not have
            the attribute `attr`.
        kind : {'min', 'max'}
            The type of optimum to search for, either 'min' or 'max'.
        style : {'branching', 'arborescence'}
            If 'branching', then an optimal branching is found. If `style` is
            'arborescence', then a branching is found, such that if the
            branching is also an arborescence, then the branching is an
            optimal spanning arborescences. A given graph G need not have
            an optimal spanning arborescence.

        Returns
        -------
        H : (multi)digraph
            The branching.

        """
        self._init(attr, default, kind, style)
        uf = self.uf

        G, B = self.G, self.B
        nodes = iter(list(G.nodes()))
        attr = self._attr

        def desired_edge(v):
            """
            Find the edge directed toward v with maximal weight.

            """
            edge = None
            weight = -INF
            for u, _, key, data in G.in_edges(v, data=True, keys=True):
                new_weight = data[attr]
                if new_weight > weight:
                    weight = new_weight
                    edge = (u, v, key, new_weight)

            return edge, weight

        def is_forest(G):
            """
            A forest is a graph with no undirected cycles.

            For directed graphs, `G` is a forest if the underlying graph is a forest.
            The underlying graph is obtained by treating each directed edge as a single
            undirected edge in a multigraph.

            """
            if len(G) == 0:
                raise nx.exception.NetworkXPointlessConcept('G has no nodes.')

            if G.is_directed():
                components = nx.weakly_connected_component_subgraphs
            else:
                components = nx.connected_component_subgraphs

            for component in components(G):
                # Make sure the component is a tree.
                if component.number_of_edges() != component.number_of_nodes() - 1:
                    return False

            return True

        def is_branching(G):
            if not is_forest(G):
                return False

            if max(G.in_degree().values()) > 1:
                return False

            return True


        def get_final_graph_and_branch(nodes, max_mem_usage):
            D = set()
            unroll = ''
            while True:
                # (I1): Choose a node v in G^i not in D^i.
                try:
                   v = next(nodes)
                except StopIteration:
                    # If there are no more new nodes to consider, then we *should*
                    # meet the break condition (b) from the paper:
                    #   (b) every node of G^i is in D^i and E^i is a branching
                    # Construction guarantees that it's a branching.
                    if len(G) != len(B):
                        raise Exception('Graph and branching must have the same number of nodes.')
                    if len(B):
                        if not(is_branching(B)):
                            raise Exception('The branching must be a branching by definition.')

                    # Add these to keep the lengths equal. Element i is the
                    # circuit at level i that was merged to form branching i+1.
                    # There is no circuit for the last level.
                    with open('circuits/'+str(self.level), 'wb') as output:
                        pickle.dump([], output)
                    self.minedge_circuit.append(None)
                    return G, B, max_mem_usage
                else:
                    if v in D:
                        #print("v in D", v)
                        continue

                # Put v into bucket D^i.
                #print("Adding node {0}".format(v))
                D.add(v)
                B.add_node(v)

                edge, weight = desired_edge(v)
                #print("Max edge is {0!r}".format(edge))
                if edge is None:
                    # If there is no edge, continue with a new node at (I1).
                    continue
                else:
                    # Determine if adding the edge to E^i would mean its no longer
                    # a branching. Presently, v has indegree 0 in B---it is a root.
                    u = edge[0]

                    if uf[u] == uf[v]:
                        # Then adding the edge will create a circuit. Then B
                        # contains a unique path P from v to u. So condition (a)
                        # from the paper does hold. We need to store the circuit
                        # for future reference.
                        Q_nodes, Q_edges = get_path(B, v, u)
                        Q_edges.append(edge[2])
                    else:
                        # Then B with the edge is still a branching and condition
                        # (a) from the paper does not hold.
                        Q_nodes, Q_edges = None, None

                    # Conditions for adding the edge.
                    # If weight < 0, then it cannot help in finding a maximum branching.
                    if not(self.style == 'branching' and weight <= 0):
                        dd = {attr: weight}
                        B.add_edge(u, v, key=edge[2], **dd)
                        G[u][v][edge[2]]['candidate'] = True
                        uf.union(u, v)
                        if Q_edges is not None:
                            #print("Edge introduced a simple cycle:")
                            #print(Q_nodes, Q_edges)

                            # Move to method
                            # Previous meaning of u and v is no longer important.

                            # Apply (I2).
                            # Get the edge in the cycle with the minimum weight.
                            # Also, save total - the incoming weights for each node.
                            minweight = INF
                            minedge = None
                            Q_incoming_weight = {}
                            for edge_key in Q_edges:
                                u, v, data = B.edge_index[edge_key]
                                w = data[attr]
                                Q_incoming_weight[v] = w
                                if  w < minweight:
                                    minweight = w
                                    minedge = edge_key


                            with open('circuits/'+str(self.level), 'wb') as output:
                                pickle.dump(Q_edges, output)
                            self.minedge_circuit.append(minedge)

                            # Now we mutate it.
                            new_node = self.template.format(self.level)

                            #print(minweight, minedge, Q_incoming_weight)

                            G.add_node(new_node)
                            new_edges = []



                            # invece di G.edges usare Q_incoming
                            for u, v, key, data in G.edges(data=True, keys=True):
                                if u in Q_incoming_weight:
                                    if v in Q_incoming_weight:
                                        # The edge must be saved to rebuild the graph
                                        dd = data.copy()
                                        if 'candidate' in dd:
                                            del dd['candidate']
                                        # unroll.append((u, v, key, dd))  
                                        unroll += str(u) + '|' + str(v) + '|' + str(key) +'|' + str(dd[self.attr]) + '\n'                                     
                                    else:
                                        # Outgoing edge. Make it from new node
                                        new_edges.append((new_node, v, key, data.copy()))
                                        # The edge must be saved to rebuild the graph
                                        dd = data.copy()
                                        if 'candidate' in dd:
                                            del dd['candidate']
                                        # unroll.append((u, v, key, dd))

                                        unroll += str(u) + '|' + str(v) + '|' + str(key) +'|' + str(dd[self.attr]) + '\n'          
                                else:
                                    if v in Q_incoming_weight:
                                        # Incoming edge. Change its weight
                                        w = data[attr]
                                        w += minweight - Q_incoming_weight[v]
                                        dd = data.copy()
                                        dd[attr] = w
                                        new_edges.append((u, new_node, key, dd))

                                        # The edge must be saved to rebuild the graph                                       
                                        dd = data.copy()
                                        if 'candidate' in dd:
                                            del dd['candidate']
                                        unroll += str(u) + '|' + str(v) + '|' + str(key) +'|' + str(dd[self.attr]) + '\n'
                                        # unroll.append((u, v, key, dd))

                                    else:
                                        # Outside edge. No modification necessary.
                                        continue

                            with open('unroll/'+str(self.level), 'w') as output:
                                output.write(unroll)
                            unroll = ''


                            current_mem_usage = psutil.virtual_memory().percent 
                            if max_mem_usage < current_mem_usage:
                                max_mem_usage = current_mem_usage
                            
                            if current_mem_usage > 89:
                                exit('Too much ram usage: '+str(current_mem_usage))


                            G.remove_nodes_from(Q_nodes)
                            B.remove_nodes_from(Q_nodes)
                            D.difference_update(set(Q_nodes))

                            for u, v, key, data in new_edges:
                                G.add_edge(u, v, key, **data)
                                if 'candidate' in data:
                                    del data['candidate']
                                    B.add_edge(u, v, key, **data)
                                    uf.union(u, v)

                            current_mem_usage = psutil.virtual_memory().percent 
                            if max_mem_usage < current_mem_usage:
                                max_mem_usage = current_mem_usage
                            
                            if current_mem_usage > 89:
                                exit('Too much ram usage: '+str(current_mem_usage))
                            
                            new_edges = []
                            nodes = iter(list(G.nodes()))
                            self.level += 1


            return None, None, max_mem_usage


        def is_root(G, u, edgekeys):
            """
            Returns True if `u` is a root node in G.

            Node `u` will be a root node if its in-degree, restricted to the
            specified edges, is equal to 0.

            """
            if u not in G:
                #print(G.nodes(), u)
                raise Exception('{0!r} not in G'.format(u))
            for v in G.pred[u]:
                for edgekey in G.pred[u][v]:
                    if edgekey in edgekeys:
                        return False, edgekey
            else:
                return True, None

        # (I3) Branch construction.
        def branch_construction(graph, branching, max_mem_usage):
            H = self.G_original_class
            # Start with the branching edges in the last level.
            edges = set(branching.edge_index)
            while self.level > 0:
                self.level -= 1

                # The current level is i, and we start counting from 0.

                # We need the node at level i+1 that results from merging a circuit
                # at level i. randomname_0 is the first merged node and this
                # happens at level 1. That is, randomname_0 is a node at level 1
                # that results from merging a circuit at level 0.
                merged_node = self.template.format(self.level)

                # The circuit at level i that was merged as a node the graph
                # at level i+1.
                with open('circuits/'+str(self.level), 'rb') as input_file:
                    circuit = pickle.load(input_file)
                 
                #print
                #print(merged_node, self.level, circuit)
                #print("before", edges)
                # Note, we ask if it is a root in the full graph, not the branching.
                # The branching alone doesn't have all the edges.
                isroot, edgekey = is_root(graph, merged_node, edges)
                
                # Rebuilding the graph of the level i
                graph.remove_node(merged_node)

                with open('unroll/'+str(self.level), 'rb') as input_file:
                    unroll = input_file.readlines()
                
                for line in unroll:
                    line = line[:-1]
                    u,v,key,data = line.split('|')
                    graph.add_edge(u,v,int(key),{self.attr: data})


                # for u,v,key,data in unroll:
                #         graph.add_edge(u,v,key,**data)

                edges.update(circuit)

                current_mem_usage = psutil.virtual_memory().percent 
                if max_mem_usage < current_mem_usage:
                    max_mem_usage = current_mem_usage
                
                if current_mem_usage > 89:
                    exit('Too much ram usage: '+str(current_mem_usage))

                if isroot:
                    minedge = self.minedge_circuit[self.level]
                    if minedge is None:
                        raise Exception

                    # Remove the edge in the cycle with minimum weight.
                    edges.remove(minedge)
                else:
                    # We have identified an edge at next higher level that
                    # transitions into the merged node at the level. That edge
                    # transitions to some corresponding node at the current level.
                    # We want to remove an edge from the cycle that transitions
                    # into the corresponding node.
                    #print("edgekey is: ", edgekey)
                    #print("circuit is: ", circuit)
                    # The branching at level i
                    
                    #print(G.edge_index)
                    target = graph.edge_index[edgekey][1]
                    for edgekey in circuit:
                        u, v, data = graph.edge_index[edgekey]
                        if v == target:
                            break
                    else:
                        raise Exception("Couldn't find edge incoming to merged node.")
                    #print("not a root. removing {0}".format(edgekey))

                    edges.remove(edgekey)

            self.edges = edges

            for edgekey in edges: 
                # print (graph.edge_index)
                u, v, d = graph.edge_index[edgekey]
                dd = {self.attr: self.trans(d[self.attr])}
                # TODO: make this preserve the key. In fact, make this use the
                # same edge attributes as the original graph.
                H.add_edge(u, v, **dd)

            # filelist = [ f for f in os.listdir('unroll') ] 
            # for f in filelist:
            #     os.remove('unroll/'+f)

            # filelist = [ f for f in os.listdir('circuits') ]
            # for f in filelist:
            #     os.remove('circuits/'+f)

            return H, max_mem_usage

        max_mem_usage = 0
        G, B, max_mem_usage = get_final_graph_and_branch(nodes, max_mem_usage)
        return branch_construction(G, B, max_mem_usage)

def is_tree(G):
    """
    Returns `True` if `G` is a tree.

    A tree is a connected graph with no undirected cycles.

    For directed graphs, `G` is a tree if the underlying graph is a tree. The
    underlying graph is obtained by treating each directed edge as a single
    undirected edge in a multigraph.
    """
    if len(G) == 0:
        raise nx.exception.NetworkXPointlessConcept('G has no nodes.')

    # A connected graph with no cycles has n-1 edges.
    if G.number_of_edges() != len(G) - 1:
        return False

    if G.is_directed():
        is_connected = nx.is_weakly_connected
    else:
        is_connected = nx.is_connected

    return is_connected(G)


def is_arborescence(G):
    """
    Returns `True` if `G` is an arborescence.

    An arborescence is a directed tree with maximum in-degree equal to 1.
    """
    if not is_tree(G):
        return False

    if max(G.in_degree().values()) > 1:
        return False

    return True

def maximum_spanning_arborescence(G, attr='weight', default=1):
    ed = Edmonds(G)
    B, max_mem_usage = ed.find_optimum(attr, default, kind='max', style='arborescence')
    if not is_arborescence(B):
        msg = 'No maximum spanning arborescence in G.'
        raise nx.exception.NetworkXException(msg)
    return B, max_mem_usage

def minimum_spanning_arborescence(G, attr='weight', default=1):
    ed = Edmonds(G)
    B = ed.find_optimum(attr, default, kind='min', style='arborescence')
    if not is_arborescence(B):
        msg = 'No maximum spanning arborescence in G.'
        raise nx.exception.NetworkXException(msg)
    return B
