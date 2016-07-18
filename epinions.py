import networkx as nx
import random
pathname = 'graphs/soc-Epinions1.txt'
out_pathname = 'graphs/epinions.txt'

def readDirectGraph(pathname):
  infile = open(pathname,'r')
  graph = nx.DiGraph()
  for line in infile:
    if "#" not in line:
      u,v = line.split()
      graph.add_edge(u, v)
  return graph

"From Trust Management for the Semantic Web"

graph = readDirectGraph(pathname)
out_file = open(out_pathname, 'w')
quality = {}
for node in graph.nodes():
	qual = random.gauss(0.5, 0.25)
	if qual < 0:
		qual = 0
	elif qual > 1:
		qual = 1
	quality[node] = qual

for node in graph.nodes():
	for neigh in graph.successors(node):
		prob = random.uniform( max( (quality[neigh] - (1-quality[node])), 0) , min( (quality[neigh] + (1-quality[node]) ), 1) )
		graph[node][neigh]['act_prob'] = prob
		out_file.write(node+'\t'+neigh+'\t'+str(prob)+'\n')
out_file.close()

