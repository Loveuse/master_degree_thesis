import networkx as nx
import random
import math
import pickle

# from pudb import set_trace; set_trace()

def joinFlows(infected_subgraph, hits, n_active_reverse_flows, active_reverse_flows):
	join_rv = {i: set() for i in infected_subgraph.nodes()}
	for u_node in active_reverse_flows:
		for node in active_reverse_flows:

		# join the flows if the reverse flow analyzed is active and they are at the same node
		# a: not the same node - b: reverse flow active on node - c: reverse flow must be started - d: condition of join
			if node!=u_node and hits[u_node][1]!=None and hits[node][1]!=node and hits[node][1] == hits[u_node][1]:
				join_rv[hits[node][1]].add(u_node)
				join_rv[hits[node][1]].add(node)

	for v_node in join_rv:
		for node in list(join_rv[v_node])[1:]:
			active_reverse_flows.remove(node)
			n_active_reverse_flows -= 1
			u_node = list(join_rv[v_node])[0]
			hits[u_node][2] = list( set().union(set(hits[u_node][2]), set(hits[node][2])) )

	return hits, n_active_reverse_flows, active_reverse_flows

def iMeter(infected_subgraph, suspects): 

	n_active_reverse_flows = len(infected_subgraph.nodes())
	hits = {i: [0, None, [i]] for i in infected_subgraph.nodes()}
	active_reverse_flows = [i for i in infected_subgraph.nodes()]
	while n_active_reverse_flows > 0:		 
		for u_node in active_reverse_flows[:]:
			# if reverse flow is active proceeds
			# if u_node in active_reverse_flows:
				# neighboors of the current node where the reverse flow is started from u_node
			neighboors = infected_subgraph.predecessors(hits[u_node][1] if hits[u_node][1] != None else u_node)
			# difference between neighboors of u and the nodes already visited from the reverse flow
			neighboors = [elem for elem in neighboors if elem not in hits[u_node][2]]

			if len(neighboors) == 0:
				active_reverse_flows.remove(u_node)
				
				# print 'No neighboors'
				n_active_reverse_flows -= 1
			else:
				# does the reverse flow stop at the current node where the reverse flow is started from u_node?
				stop_prob = 1
				for neighboor in neighboors:
					stop_prob *= (1 - infected_subgraph[neighboor][hits[u_node][1] if hits[u_node][1] != None else u_node]['act_prob'])
				# yes, it does
				if random.random() <= stop_prob:
					active_reverse_flows.remove(u_node)
					n_active_reverse_flows -= 1
				
				elif len(neighboors) == 1:
					# update where is the reverse flow started from u_node
					prev_curr = hits[u_node][1] if hits[u_node][1] != None else u_node
					hits[u_node][1] = neighboors[0]
					# update the hits number of the neighboor
					hits[neighboors[0]][0] += 1
					hits[u_node][2].append(neighboors[0])

				else:
					# order neighboors by high prob to active the current node where the reverse flow is started from u_node
					neighboors = [(neighboor, infected_subgraph[neighboor][hits[u_node][1] if hits[u_node][1] != None else u_node]['act_prob']) for neighboor in neighboors]
					neighboors = sorted(neighboors, key=lambda x: x[1], reverse=True)
					founded_neighboor = False
					while not founded_neighboor:
						for neighboor in neighboors:
							if random.random() <= neighboor[1]:
								v_node = neighboor[0] 
								hits[u_node][1] = v_node
								hits[v_node][0] += 1
								hits[u_node][2].append(v_node)
								founded_neighboor = True
								break

		hits, n_active_reverse_flows, active_reverse_flows = joinFlows(infected_subgraph, hits, n_active_reverse_flows, active_reverse_flows)

	for node in hits:
		suspects[node].append(hits[node][0]) 
	return suspects


def _iMeterSort(infected_subgraph, R, k):
	suspects = {i: [] for i in infected_subgraph.nodes()}
	for i in xrange(R):
		suspects = iMeter(infected_subgraph, suspects) 
	for i in suspects:
		suspects[i] = sum(suspects[i], 0.0)/len(suspects[i])
	return [source[0] for source in sorted(suspects.items(), key=lambda x: x[1], reverse=True)[:k]]

def IMeterSort(infected_subgraph, k=1):
	R = math.log(len(infected_subgraph.nodes()))
	sources = _iMeterSort(infected_subgraph, int(round(R)), k)
	return sources 

