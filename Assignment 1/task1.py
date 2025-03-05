import itertools, math, multiprocessing
from typing import List, Tuple

class SRFLPSolver:
    def __init__(self, input_file: str): 
        # Initialize SRFLP solver
        self.n, self.widths, self.weights = self._parse_input_file(input_file)

    def _parse_input_file(self, filename: str) -> Tuple[int, List[int], List[List[int]]]:
        # Parse input file
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        n = int(lines[0].strip())
        widths = list(map(int, lines[1].split()))
        
        # Complete symmetric weight matrix
        weights = [[0] * n for _ in range(n)]
        for i in range(n):
            row = list(map(int, lines[i+2].split()))
            for j in range(i+1, n):
                weights[i][j] = row[j-i]
                weights[j][i] = row[j-i]
        
        return n, widths, weights

    def calculate_distance(self, permutation: List[int]) -> float:
        # Calculate total travel distance for given permutation 
        # Accurate calculation according to assignment requirements
        if len(permutation) < 2:
            return 0.0

        total_cost = 0
        locations = [0] * self.n
        
        # Calculate device locations in sequence
        current_pos = 0
        for device in permutation:
            locations[device] = current_pos + self.widths[device] / 2
            current_pos += self.widths[device]
        
        # Calculate cost for all device pairs
        for i in range(len(permutation)):
            for j in range(i+1, len(permutation)):
                # Calculate distance between two devices
                device_i, device_j = permutation[i], permutation[j]
                
                # d(πi, πj) = (l_πi + l_πj)/2 + ∑(i ≤ k ≤ j) l_πk
                # Calculate midpoint between two devices
                mid_point = (locations[device_i] + locations[device_j]) / 2
                
                # Sum widths of intermediate devices
                intermediate_width = sum(
                    self.widths[permutation[k]] 
                    for k in range(i+1, j)
                )
                
                # Distance = midpoint + widths of intermediate devices
                distance = mid_point + intermediate_width
                
                # Final cost = weight * distance
                total_cost += self.weights[device_i][device_j] * distance
        
        return total_cost

    def brute_force_solve(self) -> Tuple[List[int], float]:
        # Find optimal solution using exhaustive search method
        min_cost = float('inf')
        best_permutation = None
        
        for perm in itertools.permutations(range(self.n)):
            cost = self.calculate_distance(list(perm))
            if cost < min_cost:
                min_cost = cost
                best_permutation = list(perm)
        
        return best_permutation, min_cost

    def branch_and_bound_solve(self) -> Tuple[List[int], float]:
        # Find optimal solution using branch and bound (backtracking) method
        def backtrack(current_perm: List[int], remaining: set) -> Tuple[List[int], float]:
            # Calculate cost if all devices are placed
            if not remaining:
                return current_perm, self.calculate_distance(current_perm)
            
            best_perm = None
            min_cost = float('inf')
            
            # Backtrack through remaining devices
            for device in remaining:
                new_perm = current_perm + [device]
                new_remaining = remaining.copy()
                new_remaining.remove(device)
                
                # Calculate cost of partial permutation
                candidate_perm, candidate_cost = backtrack(new_perm, new_remaining)
                
                if candidate_cost < min_cost:
                    min_cost = candidate_cost
                    best_perm = candidate_perm
            
            return best_perm or current_perm, min_cost

        # First call starts with empty permutation and full device set
        return backtrack([], set(range(self.n)))

    def parallel_brute_force_solve(self, num_processes: int = None) -> Tuple[List[int], float]:
        # Parallel exhaustive search method
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()
        
        # Divide permutations
        all_perms = list(itertools.permutations(range(self.n)))
        chunk_size = math.ceil(len(all_perms) / num_processes)
        chunks = [all_perms[i:i+chunk_size] for i in range(0, len(all_perms), chunk_size)]
        
        # Parallel processing
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(self._process_chunk, chunks)
        
        # Select optimal result
        best_perm, min_cost = min(results, key=lambda x: x[1])
        return best_perm, min_cost

    def _process_chunk(self, chunk: List[Tuple]) -> Tuple[List[int], float]:
        # Helper method to process permutation chunks
        min_cost = float('inf')
        best_perm = None
        
        for perm in chunk:
            cost = self.calculate_distance(list(perm))
            if cost < min_cost:
                min_cost = cost
                best_perm = list(perm)
        
        return best_perm, min_cost

def main():
    # Initialize problem instance solver
    solver = SRFLPSolver('Y-10_t.txt')
    
    # Exhaustive search method
    print("Brute Force Method:")
    bf_perm, bf_cost = solver.brute_force_solve()
    print(f"Optimal Permutation: {bf_perm}")
    print(f"Minimum Cost: {bf_cost}")
    
    # Parallel exhaustive search method
    print("\nParallel Brute Force Method:")
    pbf_perm, pbf_cost = solver.parallel_brute_force_solve()
    print(f"Optimal Permutation: {pbf_perm}")
    print(f"Minimum Cost: {pbf_cost}")
    
    # Branch and bound method
    print("\nBranch and Bound Method:")
    bb_perm, bb_cost = solver.branch_and_bound_solve()
    print(f"Optimal Permutation: {bb_perm}")
    print(f"Minimum Cost: {bb_cost}")

if __name__ == "__main__":
    main()