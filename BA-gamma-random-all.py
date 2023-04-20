import numpy as np
import time
import multiprocessing

# This function implements the preferential attachment algorithm 
# for generating a random graph. It takes in a list of vertexes, 
# their probabilities, and the number of edges to attach 
# to the new vertex, and returns a list of vertexes that 
# the new vertex is connected to.
def PA(vertexes, probabilities, m):
  return np.random.choice(vertexes, m, replace=False, p=probabilities)


# This function initializes the variables and data structures 
# needed for generating a Barabasi-Albert random graph. 
# It takes in  
# m - number of edges to attach to a new vertex
# N - the total number of vertexes 
# gamma - the power exponent for vertex degree
# and returns a tuple containing 
# g - the initialized graph, 
# degrees - the list of degrees of corresponding graph vertices
# degree_sum - the sum of the elements of degdeg list
# degdeg - list of corresponding degrees of vertices raised to the gamma degree
# cur - the nummber of the next new vertex.
def init(m, N, gamma):  
  g = [[] for _ in range(m + 1 + N)]
  degrees = np.zeros(m + N + 1, dtype=np.int64)
  degdeg = np.zeros(m + N + 1, dtype=np.double)

  degdeg[:m+1] = m ** gamma
  degree_sum = degdeg[0] * (m + 1)
  degrees[:m+1] = m

  for i in range(m + 1):
    for j in range(m + 1):
      if i == j:
        continue
      g[i].append(j)
  return g, degrees, degree_sum, degdeg, m + 1

# The function generates a Barabasi-Albert random graph. 
# It takes in 
# m - the number of edges to attach to a new vertex, 
# N - the total number of vertexes, 
# vertex - list of vertexes, 
# fun - the function used to calculate vertex features, 
# iters - the list of iterations at which the vertex features are calculated, 
# gamma - the power exponent for vertex degree,  
# lamb - the shape parameter for the Pareto distribution used 
#        to select the number of edges to attach to a new vertex. 
# It returns a list of vertex features calculated at the specified iterations.
def generate(m, N, vertex, fun, iters, gamma, lamb):
  g, degrees, degree_sum, degdeg, cur = init(m, N, gamma)

  k = []
  for iter_num in range(N):
    probs = degdeg / degree_sum

    if lamb > 0:
      z = 1 + min(round(np.random.pareto(lamb)), cur - 1)
    else: 
      z = m

    for v in PA(cur, probs[:cur], z):
      degrees[v]+=1
      degree_sum -= degdeg[v]
      degdeg[v] = degrees[v] ** gamma
      degree_sum += degdeg[v]
      g[v].append(cur)
      g[cur].append(v)

    degrees[cur] += z
    degdeg[cur] = z ** gamma
    degree_sum += degdeg[cur]
    
    if (cur in iters):
      for v in vertex:
        if v <= cur:
          k += [cur] + fun(ver=v, g=g, degrees=degrees, degree_sum=degree_sum)
    cur += 1
      
  return k

# The function run_thread is used to run the generate function in a separate thread. 
# It takes in 
# m - the number of edges to attach to a new vertex, 
# N - the total number of vertexes, 
# gamma - the power exponent for vertex degree, 
# lamb - the shape parameter for the Pareto distribution used 
#        to select the number of edges to attach to a new vertex, 
# fun - the function used to calculate vertex features, 
# T - the number of iterations, 
# vertex - list of vertexes, 
# iters - the list of iterations at which the vertex features are calculated, 
# i - thread index,  
# ret_dict - a dictionary used to store the results.
def run_thread(m, N, gamma, lamb, fun, T, vertex, iters, i, ret_dict):
  k = []
  for _ in range(T):
    a = generate(m, N, vertex, fun, iters, gamma, lamb)
    k += [a]
  
  ret_dict[i] = k
  
# The function runs the run_thread function in multiple threads. 
# It takes in 
# m - the number of edges to attach to a new vertex, 
# N - the total number of vertexes, 
# gamma - the power exponent for vertex degree, 
# lamb - the shape parameter for the Pareto distribution used 
#        to select the number of edges to attach to a new vertex, 
# T - the number of iterations, 
# fun - the function used to calculate vertex features, 
# vertex - list of vertexes, 
# iters - the list of iterations at which the vertex features are calculated,  
# num_workers - the number of threads to use.
def run(m, N, gamma, lamb, T, fun, vertex, iters, num_workers=2):
  
  assert T % num_workers == 0

  procs = []
  return_dict = multiprocessing.Manager().dict()


  for i in range(num_workers):
    p = multiprocessing.Process(target=run_thread, args=(m, N, gamma, lamb, fun, T//num_workers, vertex, iters, i, return_dict))
    procs.append(p)
    p.start()

  for proc in procs:
    proc.join()

  m = []
  for it in return_dict.values():
    m += it

  return m

# The function calc_s_list is used as an example of a function to calculate vertex features. 
# It takes in 
# ver - the vertex index, 
# g - the graph, 
# degrees - the list of degrees of corresponding graph vertices, 
# degree_sum - the sum of the elements of degdeg list, and 
# kwargs - optional arguments.
# This function calculates a list s for each vertex of the graph. 
# The list s contains the following elements:
# ver - vertex number
# degrees[ver] - degree of vertex
# a - total degree of all neighboring vertices
# degree_sum - total degree of all vertices in the graph
def calc_s_list(ver, g, degrees, degree_sum, **kwargs):
  a = 0
  for v in g[ver]:
    a += degrees[v]
  return [ver, degrees[ver], a, degree_sum]


# The main block of code specifies the experiment parameters and runs the run function 
# to generate a Barabasi-Albert random graph and calculate vertex features 
# at specified iterations. The results are then saved to a file.
if __name__ == '__main__':
    # calc_time()

    multiprocessing.freeze_support()

    N = 160001     # the total number of vertices in each graph
    T = 1002       # the number of graphs that need to be constructed
    M = 5          # the number of edges to attach to a new vertex 
    gamma = 0.75   # the power exponent for vertex degree
    lamb = 2.75    # the shape parameter for the Pareto distribution used 
                   # to select the number of edges to attach to a new vertex 
    vertex = [10,50,100] # list of vertexes 
    # the list of iterations at which the vertex features are calculated
    iters = [1000, 5000, 10000, 20000, 40000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000]
    
    
    res_experiment = run(M, N, gamma, lamb, T, calc_s_list, vertex, iters, num_workers=6)
    
    with open(f"BA_NPA_rand_m{M}_g{gamma}_L{lamb}_{N-1}_every20000.txt", 'w') as f3:
      for el in res_experiment:
        for el1 in el:
          f3.write(str(el1))
          f3.write(" ")
        f3.write("\n")

