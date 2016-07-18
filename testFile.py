from __future__ import division

"Debug mode on/off"
# from pudb import set_trace; set_trace()

import networkx as nx 
import random
import pickle
import os
from random import choice
import matplotlib.pyplot as plt
# from EdmondsRefact import maximum_spanning_arborescence
from EdmondsOpt import maximum_spanning_arborescence
from independent_cascade_opt import independent_cascade, getInfectedSubgraph, getInfectedCalibratedSubgraph
from imeterOpt import IMeterSort

from multiprocessing import Pool
from functools import partial

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
    print edges
    while True:
      u = random.randint(0, nodes-1)
      v = random.randint(0, nodes-1)
      if u == v or (u, v) in graph.edges():
        continue
      prob = random.random()
      graph.add_edge(u,v,act_prob=prob)
      out_file.write(str(u)+'\t'+str(v)+'\t'+str(prob)+'\n')
      edges -= 1
      print edges
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

def getInfectedOfCascadeByNode(pathname, steps, random_node, out_file):
  graph = readDirectWeightedGraph(pathname)

  out_file.write('Nodes: '+str(len(graph.nodes()))+' - Edges: '+str(len(graph.edges()))+'\n')

  out_file.write('\nSource randomly selected is '+random_node+'\n')
  infected_nodes = independent_cascade(graph, set([random_node]), steps)
  
  if len(infected_nodes) == 1:
    out_file.write('No Infected nodes. The Source is '+str(random_node)+'\n')
    return None, None, random_node, None
  else: 
    out_file.write('Number of infected: '+str(len(infected_nodes))+'\n')
  return infected_nodes, getInfectedSubgraph(graph, infected_nodes), random_node, getInfectedCalibratedSubgraph(graph, infected_nodes)

#@profile
def getInfectedOfCascade(pathname, steps, out_pathname_graph=None, nodes=None, edges=None):
  graph = readDirectWeightedGraph(pathname)

  # graph = getMyCustomGraph()
  
  if out_pathname_graph != None:
    buildGraphOnFile(out_pathname_graph, nodes, edges)
  
  #print 'Graph:'
  #print list(graph.edges_iter(data='act_prob', default=1))

  print 'Nodes: %d - Edges: %d' %(len(graph.nodes()), len(graph.edges()))

  #nx.draw(graph)
  #plt.show() 

  # random_node = choice(graph.nodes())
  
  # pageranks = nx.pagerank(graph, weight='act_prob')
  # sorted_pageranks = sorted(pageranks.items(), key=lambda x: (-x[1], x[0]))
  # index_ = random.randint(0,39)
  # random_node = sorted_pageranks[index_][0]
  # print '\nSource randomly selected is ',random_node, 'and it is the',(index_+1),'# influent node'

  # random_node = '55'
  print '\nSource randomly selected is ',random_node
  infected_nodes = independent_cascade(graph, set([random_node]), steps)
  #print infected_nodes
  
  if len(infected_nodes) == 1:
    exit('No Infected nodes. The Source is '+str(random_node))
  # elif len(infected_nodes)< 100:
    # print 'Infected nodes',infected_nodes
  else: 
    print 'Number of infected:',len(infected_nodes)
  #return infected_nodes, getInfectedSubgraph(graph, infected_nodes), random_node
  return infected_nodes, getInfectedSubgraph(graph, infected_nodes), random_node, getInfectedCalibratedSubgraph(graph, infected_nodes)


def process(steps, pathname, random_node=None, out_file=None, file_version=False):
  infected_nodes = []
  while True:
    #infected_nodes, infected_subgraph, random_node = getInfectedOfCascade(pathname, steps, out_pathname_graph)
    # infected_nodes, infected_subgraph, random_node, infected_calibrated_subgraph = getInfectedOfCascade(pathname, steps, out_pathname_graph)
    # infected_nodes, infected_subgraph, random_node, infected_calibrated_subgraph = getInfectedOfCascade(pathname, steps)
    infected_nodes, infected_subgraph, random_node, infected_calibrated_subgraph = getInfectedOfCascadeByNode(pathname, steps, random_node, out_file)
    if infected_nodes == None:
      return
    if len(infected_nodes) > 2550:
      if file_version:
        out_file.write('Too much infected nodes '+len(infected_nodes)+'\n')
      print 'Too much infected nodes', len(infected_nodes)
    else:
      break
  # print 'Infected graph'
  #print list(infected_subgraph.edges_iter(data='act_prob', default=1))

  pathname_branch = 'test_branches/branch'+random_node
  pathname_branch_cal = 'test_branches/branch_cal'+random_node

  with open(pathname_branch, 'wb') as output:
    pickle.dump(infected_subgraph, output)

  infected_subgraph = None

  branch = maximum_spanning_arborescence(pathname_branch, random_node, attr='act_prob')
  # print 'branch'
  # print list(branch.edges_iter(data='act_prob', default=1))

  with open(pathname_branch_cal, 'wb') as output:
    pickle.dump(infected_calibrated_subgraph, output)

  infected_calibrated_subgraph = None


  branch_calibrated = maximum_spanning_arborescence(pathname_branch_cal, random_node, attr='act_prob')
  # print 'branch calibrated'
  # print list(branch_calibrated.edges_iter(data='act_prob', default=1))


  # if branch.edges() == branch_calibrated.edges():
  #   print 'Same branches'
  # else:
  #   print 'Different branches'
  #   print 'Branch\n',branch.edges()
  #   print 'Branch calibrated\n',branch_calibrated.edges()

  with open(pathname_branch, 'rb') as input_file:
    infected_subgraph = pickle.load(input_file)

  if file_version:
    out_file.write('Finding the source in branch'+'\n')
  else:
    print 'Finding the source in branch'  

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

      if file_version:
        out_file.write('Source is '+ str(node)+'\n')
      else:
        print 'Source is '+ str(node)
      if node != random_node:
        if file_version:
          out_file.write('Distance from '+str(random_node)+' and '+str(node)+' is '+str(dist)+'\n')
        else:
          print 'Distance from '+str(random_node)+' and '+str(node)+' is '+str(dist)
      #exit()
      break


  if file_version:
    out_file.write('Finding the source in branch calibrated'+'\n')
  else:
    print 'Finding the source in branch calibrated'  
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

      if file_version:
        out_file.write('Source is '+ str(node)+'\n')
      else:
        print 'Source is '+ str(node)
      if node != random_node:
        if file_version:
          out_file.write('Distance from '+str(random_node)+' and '+str(node)+' is '+str(dist)+'\n')
        else:
          print 'Distance from '+str(random_node)+' and '+str(node)+' is '+str(dist)
      #exit()
      break

  if file_version:
    out_file.write('Finding the source in IMeterSort'+'\n')
  else:
    print 'Finding the source in IMeterSort'
  source = IMeterSort(infected_subgraph)

  filelist = [f for f in os.listdir('dir_Edmonds_temps')]
  
  ffornode = [f for f in filelist if 'branch'+random_node in f or 'graph'+random_node in f]
  
  for f in ffornode:
    os.remove('dir_Edmonds_temps/'+f)
        # if os.path.exists(self.temp_data):
        #     filelist = [ f for f in os.listdir(self.temp_data)]
        #     for f in filelist:
        #         os.remove(self.temp_data+'/'+f)
        # else:
        #     os.mkdir(self.temp_data)


  if file_version:
    out_file.write('Source is '+source+'\n')
  else:
    print 'Source is',source
  # print type(source),type(random_node)
  if source != random_node:
    try:
      dist = nx.shortest_path_length(infected_subgraph, source=random_node, target=source)
    except nx.NetworkXNoPath:
      dist = -1
    try:
      dist_ = nx.shortest_path_length(infected_subgraph, source=source, target=random_node)
    except nx.NetworkXNoPath:
      dist_ += 1

    if dist_ == -1 and dist == -1:
      if file_version:
        out_file.write('There isn\'t a path between '+random_node+' and '+source+'\n')
      else:
        print 'There isn\'t a path between',random_node,'and',source
    else:
      if dist_ < dist:
        dist = dist_
      if file_version:
        out_file.write('Distance from '+str(random_node)+' and '+str(source)+' is '+str(dist)+'\n')
      else:
        print 'Distance from', str(random_node),'and',str(source),'is',str(dist)
  out_file.close()
  return


"Parameters"
# steps = 4
steps = 4
n = 100
k = 40
# pathname = 'gnutella/as-caida20070917-Rand.txt'
# pathname = 'graphs/random_graph1000n2000m'
# out_pathname_graph = 'random_graph0'
pathname_graph = 'graphs/Wiki-Vote-Rand.txt'
# pathname_graph = 'gg'
# pathname_output = 'random_results/random_graph1000n2000m_steps50/s'
pathname_output = 'wiki-Vote_results/s'
#addRandomProbToGraph('gnutella/as-caida20070917.txt', 'gnutella/as-caida20070917-Rand.txt')
executions = 1
# buildGraphOnFile(pathname, 1000, 2000)
graph = readDirectWeightedGraph(pathname_graph)

filelist = [f for f in os.listdir('wiki-Vote_results')]
filelist = [f[1:] for f in filelist]
print 'filelist',len(filelist)
nodes = [node for node in graph.nodes() if node not in filelist]
print 'nodes',len(nodes)

n_nodes = len(nodes)


# node = '479'
# node = choice(graph.nodes())
# print 'random_node:',node
# process(steps, pathname_graph, random_node=str(node), out_file=open(pathname_output+str(node),'a'), file_version=True)

for node in nodes:
  print 'Process for #',n_nodes
  process(steps, pathname_graph, random_node=str(node), out_file=open(pathname_output+str(node),'a'), file_version=True)
  n_nodes -= 1
# def runProcess(node):
  # print 'Process for',node
  # for j in xrange(executions):
    # print 'Execution',j

  # process(steps, pathname_graph, random_node=str(node), out_file=open(pathname_output+str(node),'a'), file_version=True)
  # n_nodes -= 1



# stdbuf -o 0 python2 test-opt.py 2>&1 | tee result


