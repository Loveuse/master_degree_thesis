import networkx as nx 
import pickle
from Queue import PriorityQueue 
import copy
import random
import string
import sys


class MultiDiGraph_EdgeKey(nx.MultiDiGraph):
    """
    MultiDiGraph which assigns unique keys to every edge.

    Adds a dictionary edge_index which maps edge keys to (u, v, data) tuples.

    This is not a complete implementation. For Edmonds algorithm, we only use
    add_node and add_edge, so that is all that is implemented here. During
    additions, any specified keys are ignored---this means that you also
    cannot update edge attributes through add_node and add_edge.

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
        for edge in ebunch:
        	self.add_edge(*edge)

    def remove_edge_with_key(self, key):
        try:
            u, v, _  = self.edge_index[key]
            # print ('***',u,v,key)
        except KeyError:
            raise KeyError('Invalid edge key {0!r}'.format(key))
        else:
            del self.edge_index[key]
            # print ('***** self.edge_index',self.edge_index)
            self._cls.remove_edge(u, v, key)

    def remove_edges_from(self, ebunch):
        raise NotImplementedError


def random_string(L=15, seed=None):
    random.seed(seed)
    return ''.join([random.choice(string.ascii_letters) for n in range(L)])

class Camerini():

	def __init__(self, graph, Y=nx.DiGraph(), Z=nx.DiGraph(), attr='weight'):
		self.original_graph = graph
		self.attr = attr
		self._init(Y=Y, Z=Z)
		self.template = random_string()
	
	def _init(self, graph=None, Y=nx.DiGraph(), Z=nx.DiGraph()):
		self.graph = MultiDiGraph_EdgeKey()

		if graph is None:
			graph = self.original_graph

		for key, (u, v, data) in enumerate(graph.edges(data=True)):
			if (u,v) not in Z.edges():
				self.graph.add_edge(u,v,key,data.copy())
			
		for Y_edge in Y.edges(data=True):
			for (u,v) in self.graph.in_edges([Y_edge[1]]):
				if u != Y_edge[0]:
					self.graph.remove_edge(u,v)

	def best_incoming_edge(self, node, graph):
		max_weight = float('-inf')
		e = None
		# print ('Graph',graph.edges())
		for u,v,key,data in graph.in_edges([node], data=True, keys=True):
			# print ('edge',u,v,data)
			if max_weight <= data[self.attr]:
				max_weight = data[self.attr]
				e = (u,v,key,data)
		return e

	def collapse_cycle(self, graph, cycle, B, new_node):			
		for node in cycle:
			for u,v,key,data in graph.out_edges([node], data=True, keys=True):

				graph.remove_edge_with_key(key)

				if v not in cycle:
					dd = data.copy()
					graph.add_edge(new_node,v,key,**dd)		

			for u,v,key,data in graph.in_edges([node], data=True, keys=True):
				if u in cycle:
					# it will be delete later
					continue
				graph.remove_edge_with_key(key)
				dd = data.copy()
				dd_eh = list(B.in_edges([node], data=True))[0][2] 
				dd[self.attr] = dd[self.attr] - dd_eh[self.attr]
				graph.add_edge(u, new_node, key, **dd)

		for node in cycle:
			B.remove_node(node)

		return graph, B

	def add_b_to_branching(self, exposed_nodes, order, M, supernodes, B):
		v = exposed_nodes.pop(0)
		order.append(v)
		b = self.best_incoming_edge(v, M)
		if b is None:
			if v in supernodes:
				supernodes.remove(v)
			return exposed_nodes, order, M, supernodes, B, None
		b_u, b_v, b_key, b_data = b
		data = {self.attr: b_data[self.attr], 'origin': b_data['origin']}
		B.add_edge(b_u, b_v, **data)
		return exposed_nodes, order, M, supernodes, B, b

	def contracting_phase(self, B, n, supernodes, exposed_nodes, M, C, root):
		cycles = list(nx.simple_cycles(B))
		if len(cycles) > 0:
			u = 'v_'+str(n)
			supernodes.append(str(u))
			exposed_nodes.append(u)
			for node in cycles[0]:
				C[str(node)] = str(u)
			M, B = self.collapse_cycle(M, cycles[0], B, u)
			for node in B.nodes():
				if B.in_edges([node]) == []:
					if B.out_edges([node]) == []:
						B.remove_node(node)
					if node != root and node not in exposed_nodes:
						exposed_nodes.append(node)
			n += 1
		return B, n, supernodes, exposed_nodes, M, C

	def best(self, root):
		M = self.graph
		for u,v,key,data in M.edges(data=True, keys=True):
			data['origin'] = (u,v,key,{self.attr: data[self.attr]})
		n = 0
		B = nx.DiGraph()
		# C contains for every node its parent node, so it will be easy to find the path in the collapsing phase
		# from an isolated root v_1 to v_k
		nodes = M.nodes()
		if len(nodes) == 1:
			A = nx.DiGraph()
			A.add_node(nodes[0])
		C = {str(node): None for node in nodes} 
		del C[str(root)]
		exposed_nodes = [node for node in nodes]
		exposed_nodes.remove(root)
		supernodes = []
		beta = {}
		order = []

		# collapsing phase
		while len(exposed_nodes) > 0:

			exposed_nodes, order, M, supernodes, B, b = self.add_b_to_branching(exposed_nodes, order, M, supernodes, B)
			if b is None:
				continue
			b_u, b_v, b_key, b_data = b
			beta[b_v] = (b_u, b_v, b_key, b_data)
			B, n, supernodes, exposed_nodes, M, C = self.contracting_phase(B, n, supernodes, exposed_nodes, M, C, root)
		# expanding phase
		while len(supernodes) > 0:
			v_1 = supernodes.pop()
			origin_edge_v_1 = beta[v_1][3]['origin']  
			v_k = origin_edge_v_1[1]
			beta[v_k] = beta[v_1]
			v_i = str(C[str(v_k)])
			while v_i != v_1:
				supernodes.remove(v_i)
				v_i = C.pop(v_i)

		A = nx.DiGraph()
		for k, edge in beta.items():
			if k in nodes:
				u,v,key,data = edge[3]['origin']
				A.add_edge(u,v,**data.copy())

		return A

	def get_priority_queue_for_incoming_node(self, graph, v, b):
		Q = PriorityQueue()
		for u,v,key,data in graph.in_edges([v], data=True, keys=True):
			if key == b[2]:
				continue
			Q.put((-data[self.attr], (u,v,key,data)))
		return Q

	def seek(self, b, A, graph):
		v = b[1]
		Q = self.get_priority_queue_for_incoming_node(graph, v, b)
		while not Q.empty():
			f = Q.get()
			try:
				# v = T(b) is an ancestor of O(f)=f[1][1]?
				v_origin = b[3]['origin'][1]
				f_origin = f[1][3]['origin'][0] 
				nx.shortest_path(A, v_origin, f_origin)
			except nx.exception.NetworkXNoPath:
				return f[1]
		return None

	def next(self, A, Y, Z, graph=None, root='R'):
		d = float('inf')
		edge = None
		if graph is not None:
			self._init(graph)
		M = self.graph
		for u,v,key,data in M.edges(data=True, keys=True):
			data['origin'] = (u,v,key,{self.attr: data[self.attr]})
		n = 0
		B = nx.DiGraph()
		nodes = M.nodes()
		C = {str(node): None for node in nodes} 
		exposed_nodes = [node for node in nodes]
		if 'R' in exposed_nodes: 
			exposed_nodes.remove('R')
		order = []
		supernodes = []
		while len(exposed_nodes) > 0:

			exposed_nodes, order, M, supernodes, B, b = self.add_b_to_branching(exposed_nodes, order, M, supernodes, B)
			if b is None:
				continue
			b_u, b_v, b_key, b_data = b
			origin_u, origin_v = b_data['origin'][:2]
			if (origin_u, origin_v) in A.edges():
				if (origin_u, origin_v) not in Y.edges():
					f = self.seek(b, A, M)
					if f is not None:
						f_u, f_v, f_key, f_data = f
						if b_data[self.attr] - f_data[self.attr] < d:
							edge = b
							d = b_data[self.attr] - f_data[self.attr]
			B, n, supernodes, exposed_nodes, M, C = self.contracting_phase(B, n, supernodes, exposed_nodes, M, C, root)
		return edge[3]['origin'], d

	def ranking(self, k, graph=None, Y=nx.DiGraph(), Z=nx.DiGraph(), mode='branching', root='R'):
		if graph is not None:
			self._init(graph, Y, Z)

		if root == 'R' and mode == 'branching':
			best = self.best_branching
		elif root == 'R' and mode == 'arborescence_no_rooted':
			best = self.best_arborescence_no_rooted
		else:
			best = self.best_arborescence_rooted
		
		graph = self.graph.copy()
		A = best(root)  
		roots = self.find_roots(A)
		if 'R' in roots:
			roots.remove('R')

		print ('roots for ranking',roots)
		self._init(graph)
		e, d = self.next(A, Y, Z)  

		P = PriorityQueue()
		w = self.get_graph_score(A) - d if d != float('inf') else float('inf') 
		P.put( (-w, e, A, Y, Z) )

		solutions = [A]

		for j in range(1,k+1):
			w, e, A, Y, Z = P.get()
			w = -w 			

			roots.extend([root for root in self.find_roots(A) if root not in roots])
			if 'R' in roots:
				roots.remove('R')

			if w == float('-inf'):
				return solutions

			e_u, e_v, e_key, data = e
			
			Y_ = Y.copy()
			Y_.add_edge(e_u, e_v, **data.copy())
			
			Z_ = Z.copy()
			Z_.add_edge(e_u, e_v, **data.copy())

			self._init(graph, Y, Z)
			e, d = self.next(A, Y_, Z)
			w_ = self.get_graph_score(A) - d if d != float('inf') else float('inf')
			P.put( (-w_, e, A, Y_, Z) )

			self._init(graph, Y, Z_)
			Aj = best(root)
			solutions.append(Aj)
			print ('roots for Aj',self.find_roots(Aj))

			self._init(graph, Y, Z_)
			e, d = self.next(Aj, Y, Z_)
			w_ = w - d if d != float('inf') else float('inf')
			P.put( (-w_, e, Aj, Y, Z_) )

		return solutions

	def add_dummy_edges(self, root, w):
		for node in self.graph.nodes():
			while True:
				key = self.template + '_'+str(node)
				if key not in self.graph.edge_index:
					break
			self.graph.add_edge(root, node, key, act_prob=w)	

	def best_arborescence_rooted(self, root):
		return self.best(root)

	def best_arborescence_no_rooted(self, root='R'):
		self.add_dummy_edges(root, sys.float_info.min*sys.float_info.epsilon)
		return self.best(root)

	def best_branching(self, root='R'):
		self.add_dummy_edges(root, 0)
		return self.best(root)

	
	def get_graph_score(self, graph):
		score = 0
		for u,v,data in graph.edges(data=True):
			score += data[self.attr]
		return score

	def find_roots(self, branching):
		roots = []
		nodes = branching.nodes()
		if len(nodes) == 1:
			return nodes[0]
		for node in branching.nodes():
			in_neigh = branching.in_edges([node])
			if  in_neigh == [] or in_neigh == [('R', node)]:
				roots.append(node)
		return roots

	def find_roots_branching(self, k, root='R', scores=False, subgraphs=None):
		if subgraphs is not None:
			scores = {}
			if len(subgraphs) >= k:
				for subgraph in subgraphs:
					if subgraph.size() == 0:
						root_ = subgraph.nodes()[0]
						scores[root_] = {'score': 0, 'arborescence': nx.DiGraph()}
						continue
					self._init(subgraph)
					arborescence = self.best_arborescence_no_rooted(root)
					if root in arborescence.nodes():
						arborescence.remove_node(root)
					root_ = self.find_roots(arborescence)
					if len(root_) > 1:
						raise Exception('Looking for one root for ',subgraph.edges())
					root_ = root_[0]
					score = self.get_graph_score(arborescence)
					scores[root_] = {'score': score, 'arborescence': arborescence}
			else:	
				subgraphs_and_shares = [[subgraph, self.get_graph_score(subgraph), int(k/len(subgraphs)) if len(subgraph.edges()) != 0 else 1 ] for subgraph in subgraphs]
				subgraphs_and_shares = sorted(subgraphs_and_shares, key=lambda x: x[1], reverse=True)
				j = 0
				remains = k - sum([elem[2] for elem in subgraphs_and_shares])
				for i in range(remains, 0, -1):
					if len(subgraphs_and_shares[j][0].edges()) != 0:
						subgraphs_and_shares[j][2] += 1
						j += 1
				for elem in subgraphs_and_shares:
					print ('Share:',elem[2], 'len subgraph:',len(elem[0].edges()))
				scores = {}
				for elem in subgraphs_and_shares:
					print ('Elem[0] edges',len(elem[0].edges()))
					if len(elem[0].edges()) == 0:
						scores[elem[0].nodes()[0]] = {'score': 0, 'arborescence': nx.DiGraph()}
						continue
					self._init(elem[0])
					if elem[2] > 1:
						roots = set()
						while len(roots) < elem[2]:
							for root_ in roots:
								# for u,v,data in elem[0].in_edges([root_], data=True):
								# 	data[self.attr] = -1000
								# for u,v,data in elem[0].out_edges([root_], data=True):
								# 	data[self.attr] = -1000
								if root_ in elem[0].nodes():
									elem[0].remove_node(root_)
							# for node in elem[0].nodes():
							# 	for root_ in roots:
							# 		if (node, root_) not in elem[0].edges():
							# 			data = {self.attr: 10000}
							# 			elem[0].add_edge(node,root_,**data)

							self._init(elem[0])
							print ('Elem[0] components', len(list(nx.components.weakly_connected_components(self.graph))))
							branching = self.best_branching(root)
							if root in branching.nodes():
								branching.remove_node(root)
							roots_ = self.find_roots(branching)
									
							if len(roots_) == 0:
								print ('No other roots found')
							else:
								i = 0
								for root_ in roots_:
									if root_ in roots:
										print ('root',root_,' already in',roots)
										i += 1
									else:
										print ('new root',root_)
								if i == len(roots_):
									return roots_score[:k]

							roots.update(roots_)
							print ('Roots',roots,'n_roots',str(len(roots)))
							for root_ in roots:
								scores[root_] = {'descr': 'rm roots','score': 0, 'arborescence': nx.DiGraph()}

					else:
						arborescence = self.best_arborescence_no_rooted(root)
						if root in arborescence.nodes():
							arborescence.remove_node(root)
						root_ = self.find_roots(arborescence)
						if len(root_) > 1:
							raise Exception('Looking for one root for ',subgraph.edges())
						root_ = root_[0]
						score = self.get_graph_score(arborescence)
						scores[root_] = {'score': score, 'arborescence': arborescence}

		else:
			branching = self.best_branching(root)
			if root in branching.nodes():
				branching.remove_node(root)
			roots = self.find_roots(branching)
			print ('Roots in branching',roots)
			if len(roots) == 0:
				raise Exception('No roots found!')
			if len(roots) <= k and not(scores):
				return [ (root, {'score': 0, 'arborescence': nx.DiGraph()}) for root in roots]

			scores = {root: {'score': None, 'arborescence': nx.DiGraph()} for root in roots}
			for root in scores:
				self._init()
				scores[root]['arborescence'] = self.best_arborescence_rooted(root) 
				scores[root]['score'] = self.get_graph_score(scores[root]['arborescence'])


		roots_score = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
		return roots_score[:k]
