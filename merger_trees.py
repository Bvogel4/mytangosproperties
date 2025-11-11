# Merger Tree Analysis for Cosmological Simulations
# This code builds and analyzes merger trees of dark matter halos using the Tangos database
# Tangos is a database management package for organizing simulation data

from tangos.relation_finding import MultiHopMostRecentMergerStrategy
import tangos
from collections import defaultdict
import networkx as nx 
import matplotlib.pyplot as plt
import time


class HaloNode:
    """
    Represents a single halo in the merger tree.
    Each node contains information about the halo's properties and its relationships
    to other halos (progenitors) in the merger tree.
    """

    def __init__(self, halo):
        """
        Initialize a HaloNode with a halo object from the Tangos database.

        Args:
            halo: A Tangos halo object containing simulation data
        """
        self.halo = halo  # Reference to the original Tangos halo object
        self.progenitors = []  # List of HaloNode objects that are progenitors of this halo

        # Extract key physical properties from the halo
        self.ndm = halo.NDM  # Number of dark matter particles
        self.mvir = halo['Mvir']  # Virial mass (total mass within virial radius)
        #self.mstar = halo['Mstar']  # Stellar mass

        # Merger-related properties (set later during tree construction)
        self.merger_time = None  # Time when merger occurred (in Gyr)
        self.merger_ratio = None  # Mass ratio of merger event
        self.is_merger = False  # Boolean flag indicating if this halo experienced a merger

    def calculate_merger_ratio(self):
        """
        Calculate whether this halo represents a merger event and compute the merger ratio.
        A merger is defined as having multiple progenitors.
        Merger ratio = mass of main progenitor / sum of masses of other progenitors
        """
        if len(self.progenitors) > 1:
            # Multiple progenitors indicate a merger event
            self.is_merger = True

            # Find the most massive progenitor (main branch)
            main_progenitor_mass = max(p.mvir for p in self.progenitors)

            # Sum masses of all other progenitors (merging branches)
            sum_other_masses = sum(p.mvir for p in self.progenitors if p.mvir != main_progenitor_mass)

            if sum_other_masses > 0:
                self.merger_ratio = main_progenitor_mass / sum_other_masses
            else:
                # Edge case: avoid division by zero
                self.merger_ratio = None
                self.is_merger = False
        else:
            # Single or no progenitors - not a merger
            self.is_merger = False
            self.merger_ratio = None


def build_merger_tree(input_halo, max_depth=10, min_fractional_weight=0.01, min_fractional_NDM=0.01, timeout=600):
    """
    Build a complete merger tree starting from a given halo and tracing back through time.

    Args:
        input_halo: Starting halo (Tangos halo object) - typically the final/most recent halo
        max_depth: Maximum number of generations to trace back in time
        min_fractional_weight: Minimum connection weight threshold (as fraction of max weight)
        min_fractional_NDM: Minimum particle count threshold (as fraction of max NDM)
        timeout: Maximum time in seconds to spend building the tree (prevents infinite loops)

    Returns:
        tuple: (tree dictionary organized by depth, main_line list of primary progenitors)
    """
    start_time = time.time()

    # Tree structure: dictionary where keys are depth levels and values are lists of HaloNodes
    tree = defaultdict(list)

    # Create the root node (starting halo at depth 0)
    start_node = HaloNode(input_halo)
    tree[0].append(start_node)

    # Use Tangos strategy to efficiently query all potential progenitor relationships
    # This gets all halo-to-halo links within the specified depth limit
    strategy = tangos.relation_finding.MultiHopAllProgenitorsStrategy(input_halo, nhops_max=max_depth)
    link_objs = strategy._get_query_all()

    # Create a lookup cache for faster access to progenitor links
    # Key: halo ID, Value: list of link objects pointing to this halo's progenitors
    link_cache = defaultdict(list)
    for obj in link_objs:
        link_cache[obj.halo_from_id].append(obj)

    def build_subtree(node, depth):
        """
        Recursively build the merger tree by finding and processing progenitors.

        Args:
            node: Current HaloNode to find progenitors for
            depth: Current depth in the tree (0 = starting halo)

        Returns:
            HaloNode: The main progenitor (most massive) or None if no valid progenitors
        """
        # Stop recursion if we've reached maximum depth or timeout
        if depth >= max_depth or time.time() - start_time > timeout:
            return

        # Get all potential progenitor links for this halo
        link_objs = link_cache.get(node.halo.id, [])

        # Find the maximum particle count among all potential progenitors
        # Used for filtering out very small progenitors
        max_NDM = max((o.halo_to.NDM for o in link_objs), default=0)

        # Process each potential progenitor link
        for obj in link_objs:
            progenitor = obj.halo_to  # The progenitor halo

            # Find the maximum link weight for this halo (for normalization)
            max_weight = max(o.weight for o in link_objs if o.halo_from_id == obj.halo_from_id)

            # Apply filtering criteria to determine if this is a significant progenitor
            if (obj.weight > max_weight * min_fractional_weight and
                    progenitor.NDM > min_fractional_NDM * max_NDM):

                # Create a new node for this progenitor
                progenitor_node = HaloNode(progenitor)
                progenitor_node.weight = obj.weight

                # Avoid duplicate progenitors (check halo number, not object identity)
                if progenitor_node.halo.halo_number not in [p.halo.halo_number for p in node.progenitors]:
                    # Add progenitor to current node and to the tree at the next depth level
                    node.progenitors.append(progenitor_node)
                    tree[depth + 1].append(progenitor_node)

                    # Recursively build subtree for this progenitor
                    build_subtree(progenitor_node, depth + 1)

        # After processing all progenitors, analyze merger properties
        node.calculate_merger_ratio()
        if node.is_merger:
            # Record the time when this merger occurred
            node.merger_time = node.halo.timestep.time_gyr

        # Return the main progenitor (most massive) for building the main evolutionary line
        return max(node.progenitors, key=lambda x: x.mvir) if node.progenitors else None

    # Build the main evolutionary line by following the most massive progenitor at each step
    main_line = [start_node]
    current_node = start_node

    # Continue following the main branch until no more progenitors are found
    while current_node:
        current_node = build_subtree(current_node, len(main_line))
        if current_node:
            main_line.append(current_node)

    return tree, main_line


def print_merger_tree(tree):
    """
    Print a human-readable representation of the merger tree.
    Shows halo properties, progenitor relationships, and merger information.

    Args:
        tree: Dictionary representing the merger tree (from build_merger_tree)
    """
    # Iterate through each depth level in the tree
    for depth, nodes in tree.items():
        print(f"Depth {depth}:")  # Depth 0 = starting halo, higher = further back in time

        # Print information for each halo at this depth level
        for node in nodes:
            # Basic halo identification and timing
            print(f"  Halo {node.halo.halo_number} at time {node.halo.timestep.time_gyr:.2f} Gyr")

            # Physical properties
            print(f"    Mvir: {node.mvir:.2e} Msun, NDM: {node.ndm}")

            # Progenitor relationships
            if node.progenitors:
                print("    Progenitors:", [prog.halo for prog in node.progenitors])

            # Merger event information
            if node.is_merger:
                print(f"    Merger at {node.merger_time} Gyr with ratio {node.merger_ratio}")
        print()  # Blank line between depth levels


def visualize_tree(tree,main_line):
    """
    Create a visual representation of the merger tree using NetworkX and Matplotlib.

    Note: Requires networkx to be imported (currently commented out).
    Shows halos as nodes sized by mass, connected by edges showing evolutionary relationships.
    Organized vertically by cosmic time.

    Args:
        tree: Dictionary representing the merger tree (from build_merger_tree)
    """
    # Create a directed graph where edges point from progenitors to descendants
    G = nx.DiGraph()
    time_to_nodes = defaultdict(list)
    colormap = []  # For coloring nodes by merger status

    # Add nodes and edges to the graph
    for depth, nodes in tree.items():
        for node in nodes:
            # Round time for cleaner display
            time = round(node.halo.timestep.time_gyr, 2)

            color = 'pink'
            if node.is_merger:
                color = 'blue'
            #check if part of the main evolutionary line
            if node in main_line:
                color = 'red'
            if node in main_line and node.is_merger:
                color = 'purple'

            # Add node with properties for visualization
            colormap.append(color)
            G.add_node(node.halo.id, time=time, mvir=node.mvir)
            time_to_nodes[time].append(node.halo.id)

            # Add edges from progenitors to this halo
            for prog in node.progenitors:
                G.add_edge(prog.halo.id, node.halo.id)


    # Calculate positions for nodes - organize by time (vertical) and spread horizontally
    pos = {}
    times = sorted(time_to_nodes.keys(), reverse=True)  # Newest times first (top)

    for i, time in enumerate(times):
        nodes = time_to_nodes[time]
        y = 1 - (i / (len(times) - 1))  # Normalize y position (1 = top, 0 = bottom)

        # Distribute nodes horizontally at each time slice
        for j, node in enumerate(nodes):
            x = (j + 1) / (len(nodes) + 1)  # Evenly space nodes horizontally
            pos[node] = (x, y)

    # Create the plot
    plt.figure(figsize=(12, 12))

    # Draw connections between halos (evolutionary relationships)
    nx.draw_networkx_edges(G, pos, arrows=True, edge_color='gray', arrowsize=20)

    # Draw halos as circles, sized by mass
    max_mass = max(nx.get_node_attributes(G, 'mvir').values())
    node_sizes = [300 * (G.nodes[node]['mvir'] / max_mass) ** 0.5 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colormap, node_size=node_sizes)

    # Add labels showing halo ID and time
    #labels = {node: f"{node}\n{G.nodes[node]['time']:.2f} Gyr" for node in G.nodes}
    #nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Add time scale on y-axis
    y_positions = [(1 - (i / (len(times) - 1))) for i in range(len(times))]
    y_labels = [f"{time:.2f} Gyr" for time in times]
    plt.yticks(y_positions, y_labels)

    # Formatting
    plt.title("Time-Organized Merger Tree Visualization")
    plt.xlabel("Halo ID")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.show()