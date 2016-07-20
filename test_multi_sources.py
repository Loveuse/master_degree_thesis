from __future__ import division

"Debug mode on/off"
# from pudb import set_trace; set_trace()

import networkx as nx 
import random
# import pickle
from random import choice

from camerini import Camerini

from independent_cascade_opt import independent_cascade, get_infected_subgraphs
from imeterOpt import IMeterSort

import os
import csv
from timeit import default_timer as timer

from multiprocessing import Pool, Manager

def read_direct_weighted_graph(pathname):
  infile = open(pathname,'r')
  graph = nx.DiGraph()
  for line in infile:
    if "#" not in line:
      u,v,p = line.split()
      graph.add_edge(u, v, act_prob=float(p))
  return graph


def find_roots(branching):
    roots = []
    for node in branching.nodes():
      if branching.in_edges([node]) == []:
        roots.append(node)
    return roots


def adaptive_cascade(graph, random_sources, steps, interval):
  i_low = 0
  i_high = 0
  no_inf = 0
  i = 0
  while True:
    if i>35:
      return None
    if steps == 0:
      return None

    infected_nodes = independent_cascade(graph, random_sources, steps)
    

    if no_inf > 2:
      return None

    if infected_nodes == None:
      no_inf += 1
      continue 
    #print ('Len infected nodes',len(infected_nodes),i_low, i_high, steps,interval)
    if len(infected_nodes) < interval[0]:
      i_low += 1
    else:
      if len(infected_nodes) > interval[1]:
        i_high += 1
      else:
        print 'Success for',random_sources,'k',k,'interval',interval
        return infected_nodes
    i += 1

    if infected_nodes != None:      
      if i_low > 4:
        steps += 1
        i_low = 0
        i_high = 0
      if i_high > 4:
        steps -= 1
        i_low = 0
        i_high = 0


def process(steps, graph, k, interval, writer=None, file_version=False):
  while True:
    random_sources = set()
    while len(random_sources) < k:
      random_sources.add(choice(graph.nodes()))

    print ('Random sources: ',random_sources)

    infected_nodes = adaptive_cascade(graph, random_sources, steps, interval) 
    if infected_nodes != None:
      break
    else:
      print ('fail to ',k,interval) 
  print ('# infected',len(infected_nodes))
  infected_subgraphs = get_infected_subgraphs(graph, infected_nodes)
  print ('# subgraphs',len(infected_subgraphs))

  camerini = Camerini(graph, attr='act_prob')
  # print ('# nodes',len(infected_subgraphs[0].nodes()))
  
  # print ('Camerini ranking:')
  # branchings = camerini.ranking(k, infected_subgraphs[0])
  # print ('obvious roots:',camerini.find_roots(infected_subgraphs[0]))
  # branchings = camerini.ranking(k, graph, root='root')

  # for branching in branchings:
  #   print (camerini.find_roots(branching))

  
  edges = 0
  start = timer()
  solutions = camerini.find_roots_branching(k, scores=True, subgraphs=infected_subgraphs)
  time_algo = timer() - start

  for subgraph in infected_subgraphs:
    edges += len(subgraph.edges())

  sources = []
  for element in solutions:
    sources.append(element[0])

  accuracy_algo = sum([1 for node in sources if node in random_sources])

  imeter_solutions = set() 
  start = timer()
  if len(infected_subgraphs) >= k:
    for subgraph in infected_subgraphs:
      if subgraph.size() == 0:
        imeter_solutions.update(subgraph.nodes())
        continue  
      imeter_solutions.update(IMeterSort(subgraph))
  else:
    subgraphs_and_shares = [[subgraph, camerini.get_graph_score(subgraph), int(k/len(infected_subgraphs)) if len(subgraph.edges()) != 0 else 1 ] for subgraph in infected_subgraphs]
    subgraphs_and_shares = sorted(subgraphs_and_shares, key=lambda x: x[1], reverse=True)
    j = 0

    remains = k - sum([elem[2] for elem in subgraphs_and_shares])
    for i in range(remains, 0, -1):
      if len(subgraphs_and_shares[j][0].edges()) != 0:
        subgraphs_and_shares[j][2] += 1
        j += 1
    for elem in subgraphs_and_shares:
      if elem[0].size() == 0:
        imeter_solutions.update(elem[0].nodes())
        continue 
      imeter_solutions.update(IMeterSort(elem[0], k=elem[2]))
  time_IMeterSort = timer() - start
  print ('IMeter solution',imeter_solutions)
  accuracy_IMeterSort = sum([1 for node in imeter_solutions if node in random_sources])

  writer.writerow(list(random_sources) + [len(infected_nodes), edges] + sources + [accuracy_algo, time_algo] + list(imeter_solutions) + [accuracy_IMeterSort, time_IMeterSort])

  
  return


"Parameters"
# pathname = 'gnutella/as-caida20070917-Rand.txt'
# pathname = 'graphs/random_graph1000n2000m'
# out_pathname_graph = 'random_graph0'
# pathname_graph = 'graphs/Wiki-Vote-Rand.txt'
k = 4
steps = 2

pathname_graph = 'graphs/Wiki-Vote-Rand.txt'
# pathname_graph = 'gg'
# pathname_output = 'wiki-Vote_results_multi_sources/test0_steps3.csv'
# pathname_output = 'wiki-Vote_results/s'
#addRandomProbToGraph('gnutella/as-caida20070917.txt', 'gnutella/as-caida20070917-Rand.txt')

graph = read_direct_weighted_graph(pathname_graph)

def run(*args):
  steps, graph, k, queue, interval = list(args)[0]
  i = queue.get()
  pathname = 'wiki-Vote_results_multi_sources/test_'+str(i)+'_k_'+str(k)+'.csv'

  out_file = open(pathname,'a')
  writer = csv.writer(out_file, delimiter=',')
  if os.stat(pathname).st_size == 0:
    heading = ['Sources'] + ['']*(k-1) + ['Num_Infected_Nodes', 'Num_Edges', 'Algo'] + ['']*(k-1) + ['Accuracy_Algo','Time_Algo', 'IMeterSort'] + ['']*(k-1) + ['Accuracy_IMeterSort','Time_IMeterSort']
    writer.writerow(heading)
  process(steps, graph, k, interval, writer=writer)
  queue.put(i)

n_jobs = 30
processes = 16

intervals = [100, 250, 500, 650, 1000, 1200, 1500, 1700, 2100, 2700]
queue = Manager().Queue()
  for i in range(n_jobs):
    queue.put(i)
for k in range(2,5): 
  for interval in range(0, len(intervals), 2):
   

    pool = Pool(processes)
    pool.map(run, [(steps, graph, k, queue, intervals[interval:interval+2]) for i in range(n_jobs)])
  # for process in processes:
  #   process.start()

  # for process in processes:
  #   process.join()
  # # Parallel(n_jobs=-1)(delayed(run)(steps, graph, k, i) for i in range(n_jobs) )

  # process(steps, graph, random_sources, k, writer=writer)

