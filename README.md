# mycelium-intelligence

Mushroom mycelium is one of the oldest forms of life on Earth, with millennia of evolutionary advantages yet to be explored. Recent discoveries of mycelium intelligence and demonstrations of complex adaptive behaviours have been exciting the scientific world. Fukasawa et al. (2024) found that fungal mycelium can “recognize” the difference in spatial arrangement as mycelium maintained its “X” and “O” spatial structure with growth. This display of spatial awareness compounds upon Money (2021)’s in-depth study of fungal hyphae’s extreme sensitivity to their environment, allowing them to detect surface ridges, repair wounds, and other expressions of decentralized biological decision-making. These two precedents support the viability of our research that seeks to challenge the traditional centralized “predictive brain” of current artificial intelligences with a decentralized, self-organizing collective of “node-level brains” grounded in Pleurotus ostreatus’ mycelium hyphal activity traversing mazes. While traditional artificial intelligences typically use a static, centralized, high-dimensional map of the subject to predict outcomes, our project seeks to use a dynamic graph rewriting learned from the biological growth patterns of Pleurotus ostreatus mycelium. By mapping the weights of hyphal branches as mycelium navigates mazes, we modeled individual, smaller nodular NNs, then connected the pre-trained nodes to emulate collective intelligence. Similar to our biological brain, greater neural activity creates neural pathways with heavier weights and faster signals. The same is reflected in the hyphal network of mycelium systems. The success of our project will allow for simultaneous computations, saving energy and solving complex heuristics challenging for energy-intensive, centralized architectures like large language models.

# Protocol

Materials: Still-air box, 70% ethanol, malt extract agar dishes, Pleurotus ostreatus strain from Liquid Fungi, 3D-printed maze inserts, medical gloves, computer

MYCELIUM PROTOCOL:
Sterilize counter space, still-air box, apparatuses, and the outside of containers completely with 70% ethanol.
In the still air-box, inoculate liquid mycelium spawns by dropping 3-4 drops on the center of the malt extract agar dishes. Repeat this for 3 agar plates.
Construct a maze insert that can be easily overlayed on top of the malt extract agar to create mini maze petri dishes.
Sterilize maze inserts by soaking them in ethanol for at least 15 minutes.
With a sterilized scalpel, cut two pieces of agar that have been completely inoculated with mycelium. Place one at each of the two maze endpoints.
Film growth with a time-lapse camera or document growth daily with a phone camera until mycelium growth stops (determined by the lack of growth for 5+ days).

Colonization progress of the starter batch of Pleurotus ostreatus mycelium on malt extract agar will be checked every two days. Data of the maze dishes will be recorded every day for two weeks after relocation of the fully inoculated mycelium.

COMPUTATIONAL PROTOCOL:
Observe the mycelium growth through the recorded time-lapse videos, then determine the timestamps when modeling the neural network.
Collect the images between each timestamp, and identify the nodes and edges from the mycelium’s geometry. The node should include the information of its coordinates and its connected edges. The edge should include its weight based on its thickness.
Create a dataset to train a small neural network to approximate the behavior of each node. The dataset for a single node should consist of: the weights of the connected edges, the estimated distance to the target at the node, the node's coordinates, and the difference between the current state and the conditions a timestamp later.
Determine the appropriate neural network structure based on the dataset. The model should determine whether the node will grow or deconstruct. This neural network should not be large or require extensive calculations.
Train the node-level neural network.
Create a virtual maze, and emulate the mycelium’s behavior using the trained node-level neural network combined with the dynamic graph system.
Test several mazes of different sizes, complexities, and objectives, then observe the behavior. 
