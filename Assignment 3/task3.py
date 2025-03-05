import numpy as np
import time
import sys
from collections import defaultdict

class PageRank:

    # Initialize PageRank algorithm for a web graph.
    def __init__(self, graph_file, damping_factor=0.85, epsilon=1e-6, max_iterations=100):
        
        # Read graph from file and create adjacency list
        self.graph = defaultdict(list)
        self.incoming_graph = defaultdict(list)
        self.nodes = set()
        
        # Graph statistics
        self.total_nodes = 0
        self.total_edges = 0
        
        # Read graph and compute statistics
        self._read_graph(graph_file)
        
        # PageRank specific parameters
        self.damping_factor = damping_factor
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        
        # Outgoing edge count for each node
        self.out_degree = {node: len(self.graph[node]) for node in self.nodes}
    
    # Read graph from text file, tracking nodes and edges.
    def _read_graph(self, graph_file):
        try:
            with open(graph_file, 'r') as f:
                for line in f:

                    # Skip comment lines
                    if line.startswith('#'):
                        continue
                    
                    # Parse node connections
                    parts = line.strip().split()
                    if len(parts) == 2:
                        from_node, to_node = map(int, parts)
                        self.graph[from_node].append(to_node)
                        self.incoming_graph[to_node].append(from_node)
                        self.nodes.update([from_node, to_node])
            
            self.total_nodes = len(self.nodes)
            self.total_edges = sum(len(edges) for edges in self.graph.values())
            
            print(f"Total nodes: {self.total_nodes}")
            print(f"Total edges: {self.total_edges}")

        except FileNotFoundError:
            print(f"Error: File {graph_file} not found.")
            sys.exit(1)

        except Exception as e:
            print(f"Error reading graph file: {e}")
            sys.exit(1)
    
    # Compute PageRank using specified method.
    def compute_pagerank(self, method='data_driven'):
        
        # Initialize PageRank scores uniformly
        pagerank = {node: 1.0 / self.total_nodes for node in self.nodes}
        
        start_time = time.time()
        
        # Tracking iteration details
        iteration_details = {
            'total_iterations': 0,
            'convergence_details': [],
            'computation_time': 0
        }
        
        for iteration in range(self.max_iterations):
            prev_pagerank = pagerank.copy()
            max_change = 0
            
            # Nodes to update based on method
            nodes_to_update = self.nodes if method == 'data_driven' else \
                set(node for node in self.nodes if prev_pagerank[node] > 0)
            
            for node in nodes_to_update:
                # Teleportation component
                pr_value = (1 - self.damping_factor) / self.total_nodes
                
                # Link-based component
                for incoming_node in self.incoming_graph[node]:
                    pr_value += (self.damping_factor * 
                                 prev_pagerank[incoming_node] / 
                                 max(len(self.graph[incoming_node]), 1))
                
                # Update PageRank and track change
                pagerank[node] = pr_value
                max_change = max(max_change, 
                                 abs(pr_value - prev_pagerank[node]))
            
            # Store convergence details
            iteration_details['convergence_details'].append(max_change)
            iteration_details['total_iterations'] = iteration + 1
            
            # Check convergence
            if max_change < self.epsilon:
                break
        
        # Compute total computation time
        iteration_details['computation_time'] = time.time() - start_time
        
        return pagerank, iteration_details
    
    # Detailed analysis of PageRank computation.
    def print_pagerank_analysis(self, pagerank, iteration_details, method):
        print(f"\n{method.replace('_', ' ').title()} PageRank Analysis:")
        print(f"Total Iterations: {iteration_details['total_iterations']}")
        print(f"Computation Time: {iteration_details['computation_time']:.2f} seconds")
        
        # Convergence analysis
        convergence_details = iteration_details['convergence_details']
        print("\nConvergence Details:")
        print(f"Final Change: {convergence_details[-1]}")
        print(f"Max Change: {max(convergence_details)}")
        print(f"Min Change: {min(convergence_details)}")
        
        # Top pages
        sorted_pages = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 10 Pages:")
        for rank, (page, score) in enumerate(sorted_pages[:10], 1):
            print(f"{rank}. Page {page}: {score:.6f}")
        
        # Compute additional statistics
        pr_values = list(pagerank.values())
        print("\nPageRank Statistics:")
        print(f"Mean PageRank: {np.mean(pr_values):.6f}")
        print(f"Median PageRank: {np.median(pr_values):.6f}")
        print(f"Std Deviation: {np.std(pr_values):.6f}")

def main():
    graph_file = 'web-BerkStan.txt'
    
    # Create PageRank instance with refined parameters
    pr = PageRank(graph_file, 
                  damping_factor=0.85, 
                  epsilon=1e-6,    # More precise convergence
                  max_iterations=100)
    
    # Compute and analyze PageRank for both methods
    for method in ['data_driven', 'topology_driven']:
        pagerank, iteration_details = pr.compute_pagerank(method)
        pr.print_pagerank_analysis(pagerank, iteration_details, method)

if __name__ == "__main__":
    main()