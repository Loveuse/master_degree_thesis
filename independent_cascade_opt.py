import copy
import networkx as nx
import random

# from pudb import set_trace; set_trace()


def independent_cascade(G, seeds, steps=0):
  """
  References
  ----------
  [1] David Kempe, Jon Kleinberg, and Eva Tardos.
      Influential nodes in a diffusion model for social networks.
      In Automata, Languages and Programming, 2005.
  """
  if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
      raise Exception("independent_cascade() is not defined for graphs with multiedges.")

  # make sure the seeds are in the graph
  for s in seeds:
    if s not in G.nodes():
      raise Exception("seed", s, "is not in graph")

  # change to directed graph
  if not G.is_directed():
    DG = G.to_directed()
  else:
    DG = copy.deepcopy(G)

  # init activation probabilities
  for e in DG.edges():
    #if 'act_prob' not in DG[e[0]][e[1]]:
    #  DG[e[0]][e[1]]['act_prob'] = 0.1
    if float(DG[e[0]][e[1]]['act_prob']) > 1:
      print (e)
      raise Exception("edge activation probability:", DG[e[0]][e[1]]['act_prob'], "cannot be larger than 1")

  # perform diffusion
  A = copy.deepcopy(seeds)  # prevent side effect
  #if steps <= 0:
    # perform diffusion until no more nodes can be activated
  #  return _diffuse_all(DG, A)
  # perform diffusion for at most "steps" rounds
  return _diffuse_k_rounds(DG, A, steps)


def _diffuse_k_rounds(G, A, steps):
  tried_edges = set()
  while steps > 0 and len(A) < len(G):
    len_old = len(A)
    (A, tried_edges) = _diffuse_one_round(G, A, tried_edges)
    # print (tried_edges)
    if len(A) == len_old:
      break
    steps -= 1
  return A

def _diffuse_one_round(G, A, tried_edges):
  activated_nodes_of_this_round = set()
  for s in A:
    for nb in G.successors(s):
      if nb in A or (s, nb) in tried_edges:
        continue
      if _prop_success(G, s, nb):
        activated_nodes_of_this_round.add(nb)
        #print nb
      tried_edges.add((s, nb))
  A = A.union(activated_nodes_of_this_round)
  return A, tried_edges


def _prop_success(G, src, dest):
  return random.random() <= G[src][dest]['act_prob']

def getInfectedSubgraph(graph, infected_nodes):
  subgraph = nx.DiGraph()
  for node in infected_nodes:
    for out_node in graph.out_edges([node]):
      if out_node[1] in infected_nodes:
        subgraph.add_edge(node, out_node[1], act_prob=graph[node][out_node[1]]['act_prob'])

  return subgraph 

def get_infected_subgraph(graph, infected_nodes):
  subgraph = nx.DiGraph()
  for node in infected_nodes:
    singleton = True
    for out_node in graph.out_edges([node]):
      if out_node[1] in infected_nodes:
        subgraph.add_edge(node, out_node[1], act_prob=graph[node][out_node[1]]['act_prob'])
        singleton = False
    if singleton:
        subgraph.add_node(node)
  return subgraph 


def get_infected_subgraphs(graph, infected_nodes):
  infected_graph = get_infected_subgraph(graph, infected_nodes)
  subgraphs = []
  components = list(nx.components.weakly_connected_components(infected_graph))
  if len(components) == 1:
      subgraphs.append(infected_graph)
      return subgraphs
  for comp in components:
    subgraph = nx.DiGraph()
    for node in comp:
        singleton = node
        for out_node in infected_graph.out_edges([node]):
            if out_node[1] in comp:
                subgraph.add_edge(node, out_node[1], act_prob=infected_graph[node][out_node[1]]['act_prob'])
    if subgraph.size() == 0:
        subgraph.add_node(singleton)
    subgraphs.append(subgraph)
  return subgraphs


def getInfectedCalibratedSubgraph(graph, infected_nodes):
  subgraph = nx.DiGraph()

  for node in infected_nodes:
    for out_node in graph.successors(node):
      if out_node in infected_nodes:
        prob = 1
        for neighboor in graph.successors(node):
          if neighboor not in infected_nodes:
            prob *= (1-graph[node][neighboor]['act_prob'])
        subgraph.add_edge(node, out_node, act_prob=graph[node][out_node]['act_prob']*prob)
  
  return subgraph 

