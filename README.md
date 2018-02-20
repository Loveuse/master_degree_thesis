# Master's degree thesis 
Thesis title: "Identifying the source of false information in social networks". 

Related publication: ["Contrasting the Spread of Misinformation in Online Social
Networks"](http://www.aamas2017.org/proceedings/pdfs/p1323.pdf) 


## Project description
* EdmondsMemoryOpt.py - Customised version of the **Edmonds**' algorithm with less memory/time usage compared to networkx
* EdmondsLowMemory.py - Version that uses the disk space to store intermediate results
* imeterOpt.py - **Imeter-Sort** algorithm included in the publication *"Sources of misinformation in online social networks: Who to suspect?"*, Nam P. Nguyen Dung T. Nguyen and My T. Thai. S
* independent_cascade_opt.py - An implementation of the independent cascade model
* camerini.py - Implementation of the algoritm included in the publication *"The k best spanning arborescences of a network"*,  L. Fratta P. M. Camerini and F. Maffioli (1980)
* test.py - Test class, compares Imeter-Sort with the Edmonds' algorithm accuracy under the assumption that there is only one source of false information
* test_multi_sources.py - Test class, compares Imeter-Sort with Camerini's algorithm (with an euristic applied which deals also with not connected graphs) whereas there are multiple source of false information

## Addition resources:
* thesis.pdf 
* Presentation.pdf
