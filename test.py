from __future__ import division
# from pudb import set_trace; set_trace()

import networkx as nx 
import random
import pickle
from EdmondsMemoryOpt import maximum_spanning_arborescence
from independent_cascade_opt import independent_cascade, getInfectedSubgraph, getInfectedCalibratedSubgraph
from imeterOpt import IMeterSort

from camerini import Camerini

import csv
from timeit import default_timer as timer
import os



def addRandomProbToGraph(in_pathname, out_pathname):
  input_file = open(in_pathname, 'r')
  out_file = open(out_pathname, 'w')
  for line in input_file:
    if "#" not in line:
      line_split = line.split('\t')
      out_file.write(line_split[0]+'\t'+line_split[1]+'\t'+str(random.random())+'\n')
    else:
      out_file.write(line)
  input_file.close()
  out_file.close()

def readDirectWeightedGraph(pathname):
  infile = open(pathname,'r')
  graph = nx.DiGraph()
  for line in infile:
    if "#" not in line:
      u,v,p = line.split()
      graph.add_edge(u, v, act_prob=float(p))
  return graph

def makeGraphComplete(graph):
  for u in graph.nodes():
    for v in graph.nodes():
      if (u,v) not in graph.edges():
        graph.add_edge(u,v,act_prob=0) 
  return graph

def buildGraphOnFile(out_pathname_graph, nodes, edges):
  out_file = open(out_pathname_graph, 'w')
  graph = nx.DiGraph()
  while edges > 0:
    while True:
      u = random.randint(0, nodes-1)
      v = random.randint(0, nodes-1)
      if u == v or (u, v) in graph.edges():
        continue
      prob = random.random()
      graph.add_edge(u,v,act_prob=prob)
      out_file.write(str(u)+'\t'+str(v)+'\t'+str(prob)+'\n')
      edges -= 1
      break
  out_file.close()
  return 

def getMyCustomGraph():
  graph = nx.DiGraph()
  graph.add_edge(0, 1, act_prob=0.5)
  graph.add_edge(0, 2, act_prob=0.8)
  graph.add_edge(1, 2, act_prob=0.3)
  graph.add_edge(1, 3, act_prob=0.4)
  graph.add_edge(2, 4, act_prob=0.6)
  graph.add_edge(3, 4, act_prob=0.2)
  graph.add_edge(4, 0, act_prob=0.7)

  graph.add_edge(n, 0, act_prob=random.random())
  for j in xrange(k):
     graph.add_edge(i,i+j,act_prob=random.random())
  return graph

def getInfectedOfCascadeByNode(pathname, steps, random_node):
  graph = readDirectWeightedGraph(pathname)

  infected_nodes = independent_cascade(graph, set([random_node]), steps)
  
  if len(infected_nodes) == 1:
    return None, None, random_node
  # return infected_nodes, getInfectedSubgraph(graph, infected_nodes), random_node, getInfectedCalibratedSubgraph(graph, infected_nodes)
  return infected_nodes, getInfectedSubgraph(graph, infected_nodes), random_node


def adaptive_cascade(pathname, steps, random_node, interval, writer=None):
  i_low = 0
  i_high = 0
  no_inf = 0
  while True:

    if steps == 0:
      return None, None, None

    infected_nodes, infected_subgraph, random_node = getInfectedOfCascadeByNode(pathname, steps, random_node)
    if len(infected_nodes) < interval[0]:
      i_low += 1
    else:
      if len(infected_nodes) > interval[1]:
        i_high += 1
      else:
        return infected_nodes, infected_subgraph, random_node
    
    if no_inf > 2:
      return None, None, None

    if infected_nodes == None:
      no_inf += 1
    else:      
      print ('Len infected nodes',len(infected_nodes),i_low, i_high, steps)
      if i_low > 4:
        steps += 1
        i_low = 0
        i_high = 0
      if i_high > 4:
        steps -= 1
        i_low = 0
        i_high = 0





def run_exp(steps, pathname, interval, random_node=None, writer=None, file_version=False):
  infected_nodes = []
  infected_nodes, infected_subgraph, random_node = adaptive_cascade(pathname, steps, random_node, interval)
  if infected_nodes == None:
    writer.writerow([random_node, 1, 0])
    return False
  
  start = timer()
  branch, max_mem_usage = maximum_spanning_arborescence(infected_subgraph, attr='act_prob')
  time_algo = timer()-start
  # print 'branch'
  # print list(branch.edges_iter(data='act_prob', default=1))

  # branch_calibrated = maximum_spanning_arborescence(infected_calibrated_subgraph, attr='act_prob')
  # print 'branch calibrated'
  # print list(branch_calibrated.edges_iter(data='act_prob', default=1))


  # if branch.edges() == branch_calibrated.edges():
  #   print 'Same branches'
  # else:
  #   print 'Different branches'
  #   print 'Branch\n',branch.edges()
  #   print 'Branch calibrated\n',branch_calibrated.edges()

  dist = 0
  for node in infected_nodes:
    source = True
    for edge in branch.edges():
      if node == edge[1]:
        source = False
    if source:
      dist = nx.shortest_path_length(infected_subgraph,source=random_node,target=node)
      dist_ = nx.shortest_path_length(infected_subgraph,source=node,target=random_node)
      if dist_ < dist:
        dist = dist_

  source = node
  dist_algo = dist

  start = timer()
  source_IMeterSort = IMeterSort(infected_subgraph)[0]
  time_IMeterSort = timer() - start

  dist_IMeterSort = 0
  if source_IMeterSort != random_node:
    try:
      dist_IMeterSort = nx.shortest_path_length(infected_subgraph, source=random_node, target=source_IMeterSort)
    except nx.NetworkXNoPath:
      dist_IMeterSort = -1
    try:
      dist_ = nx.shortest_path_length(infected_subgraph, source=source_IMeterSort, target=random_node)
    except nx.NetworkXNoPath:
      dist_ += 1

    if dist_ == -1 and dist_IMeterSort == -1:
          dist_IMeterSort = -1
    else:
      if dist_ < dist_IMeterSort:
        dist_IMeterSort = dist_
    
  writer.writerow([random_node, len(infected_nodes), len(infected_subgraph.edges()), source, dist_algo, time_algo, max_mem_usage, source_IMeterSort, dist_IMeterSort, time_IMeterSort])
  return True


"Parameters"
# pathname = 'gnutella/as-caida20070917-Rand.txt'
# pathname = 'graphs/random_graph1000n2000m'
pathname_graph = 'graphs/Wiki-Vote-Rand.txt'
# pathname_graph = 'graphs/epinions.txt'
steps = 2
# pathname_graph = 'graphs/random_graph200n10000m'
# pathname_output = 'random_results/random_graph200n10000m_steps3/s'
# pathname_output = 'wiki-Vote_results/s'
# pathname_output = 'random_results/single_source/random_graph200n10000m_steps3/test.csv'
pathname_output = 'wiki-vote-results/csv/test_0_step3.csv'
# pathname_output = 'epinions_results/test_0_step3.csv'
#addRandomProbToGraph('gnutella/as-caida20070917.txt', 'gnutella/as-caida20070917-Rand.txt')


#graph = readDirectWeightedGraph(pathname_graph)
graph = nx.read_edgelist(pathname_graph, create_using=nx.DiGraph(), data=(('act_prob',float),), comments='#')


out_file = open(pathname_output,'a')
writer = csv.writer(out_file, delimiter=',')
if os.stat(pathname_output).st_size == 0:
  heading = ['Source', 'Num_Infected_Nodes', 'Num_Edges', 'Algo', 'Dist_Algo', 'Time_Algo', 'Mem_Usage', 'IMeterSort', 'Dist_IMeterSort', 'Time_IMeterSort']
  writer.writerow(heading)

intervals = [100, 250, 500, 650, 1000, 1200, 1500, 1700, 2400, 2700]
n_exp = 20
evaluated = []
for i in range(0, len(intervals), 2):
  j = 0
  while j < n_exp:
    node = random.choice(graph.nodes())
    print node
    if node in evaluated:
      continue
    evaluated.append(node)

    res = run_exp(steps, pathname_graph, intervals[i:i+2], random_node=str(node), writer=writer, file_version=True)
    if res:
      j += 1


out_file.close()

