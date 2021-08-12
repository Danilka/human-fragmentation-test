from multiprocessing import Pool
import math
import numpy as np
import random
from timer_cm import Timer
from scipy.stats import truncnorm
from statistics import mean


class Simulator:

    PROCESSES = 6

    CLUSTERS = 64  # *256
    NODES_PER_CLUSTER = 16
    NODES = CLUSTERS * NODES_PER_CLUSTER
    BUSINESSES_PERCENT = 0.05  # float % from 0 to 1

    TRANSACTIONS_P2P_PERCENT = 0.2  # float % from 0 to 1
    TRANSACTIONS_P2B_PERCENT = 0.8  # float % from 0 to 1
    TRANSACTIONS_P2P_UNIQUE = 50  # Number of people that any person interacts with
    TRANSACTIONS_P2B_UNIQUE = 50  # Number of businesses that any person interacts with
    TRANSACTIONS_PER_PERSON_AVG = 1000
    TRANSACTIONS_PER_PERSON_MIN = 500
    TRANSACTIONS_PER_PERSON_MAX = 5000
    TRANSACTION_SIZE_MIN = 0.001  # float % from net worth [0; 1]
    TRANSACTION_SIZE_MAX = 0.1  # float % from net worth [0; 1]

    NET_WORTH_AVG = 1000 * 100  # In cents
    NET_WORTH_MIN = 15 * 100  # In cents
    NET_WORTH_MAX = 1000 * 1000 * 100  # In cents

    # Bills per person
    BILLS_PP_AVG = 10
    BILLS_PP_MIN = 2
    BILLS_PP_MAX = 1000

    # X and Y field coordinate boundaries.
    MAX_X = 1000
    MAX_Y = MAX_X

    # What is considered to be a close distance between nodes.
    CLOSE_DISTANCE_RADIUS = (MAX_X + MAX_Y) * 0.01  # 1% of the side.

    def __init__(self):
        # {node id} -> (x, y)
        # Coordinates are saved in float
        self.nodes_loc = {}

        # {bill_id} -> {size: 123, owner: node_id, cluster: cluster_id}
        self.bills = {}

        # {owner_id} -> [bill_id, bill_id, ...]
        self.wallets = {}

        # Set of all nodes that are businesses.
        # (business_id, business_id, business_id, ...)
        self.businesses = set()

        # Set of all nodes that are not businesses.
        # (private_node_id, private_node_id, private_node_id, ...)
        self.non_businesses = set()

        # Business receivers for each node to address.
        # {node_id} -> [business_id (node), business_id, business_id, ...]
        self.b_receivers = {}

        # Private receivers for each node to address.
        # {node_id} -> [person_id (node), person_id, person_id, ...]
        self.p_receivers = {}

        with Timer('Generate nodes'):
            self.generate_nodes()

        with Timer('Generate payees'):
            # Non parallel version
            # payees = []
            # for i in range(NODES):
            #     payees.append(generate_node_payees(i))

            # nodes_buckets = np.array_split(range(self.NODES), self.PROCESSES)
            with Pool(self.PROCESSES) as pool:
                payees = pool.map(self.generate_node_payees, range(self.NODES))

            for i in range(self.NODES):
                self.b_receivers[i] = payees[i][0]
                self.p_receivers[i] = payees[i][1]

        with Timer('Generate bills'):
            # TODO Parallelize this.
            self.generate_bills()

        self.system_status()

    def system_status(self):
        print("{} nodes.".format(len(self.nodes_loc)))
        print("Mean friends per person: {}".format(mean([len(self.p_receivers[i]) for i in range(self.NODES)])))
        print("Mean businesses per person: {}".format(mean([len(self.b_receivers[i]) for i in range(self.NODES)])))

        totals = [self.bills[x]['size'] for x in self.bills.keys()]
        print("{}$ in the system.".format(sum(totals)/100))
        print("Mean bill size ${}".format(mean(totals)/100))
        print("Total bills: {}".format(len(self.bills.keys())))
        print("Avg bills per person: {}".format(len(self.bills.keys())/self.NODES))

        wealth = [sum([self.bills[x]['size'] for x in self.wallets[i]]) for i in range(self.NODES)]
        print("Mean wealth per person ${}".format(mean(wealth)/100))
        print("Max wealth: ${}".format(max(wealth)/100))
        print("Min wealth: ${}".format(min(wealth)/100))

    def close_dist(self, distance):
        """Returns True if the distance is close."""
        return distance <= self.CLOSE_DISTANCE_RADIUS

    def far_dist(self, distance):
        """Returns True if the distance is far."""
        return not self.close_dist(distance)

    def get_asymmetric_norm(self, low, mid, upp):
        """Get one random number from an unbalanced normal distribution."""
        # if random.uniform(0, 1) < 0.5:
        if random.uniform(low, upp) < mid:
            # Less than a mean
            upp = mid
        else:
            # Greater than a mean
            low = mid

        sd = (upp - low) / 3
        distribution = truncnorm((low - mid) / sd, (upp - mid) / sd, loc=mid, scale=sd)
        return distribution.rvs().round().astype(int)

    def node_to_cluster(self, node_id: int) -> int:
        """Get cluster_id that node_id belongs to."""
        return node_id % self.NODES_PER_CLUSTER

    def distance_between_nodes(self, node_id_from: int, node_id_to: int) -> float:
        """Get distance between two node IDs."""
        return math.sqrt(
            (self.nodes_loc[node_id_from][0] - self.nodes_loc[node_id_to][0])**2
            + (self.nodes_loc[node_id_from][1] - self.nodes_loc[node_id_to][1])**2
        )

    def generate_nodes(self):

        # Generate nodes.
        for i in range(self.NODES):
            # Empty wallets are created by default.

            # Set location.
            loc = (float(random.randrange(self.MAX_X)), float(random.randrange(self.MAX_Y)))
            self.nodes_loc[i] = loc

            # Pick if the node is a business.
            if random.randrange(100) <= 100 * self.BUSINESSES_PERCENT:
                self.businesses.add(i)
            else:
                self.non_businesses.add(i)

        # Set defaults for randomized businesses and non businesses.
        self.randomized_businesses = list(self.businesses)
        self.randomized_non_businesses = list(self.non_businesses)


    def generate_node_payees(self, node_id):
        """
        Generate lists of close businesses and friends for the node_id node.
        :param node_id: Which node to generate payees for.
        :return: (list of business payees, list of private payees)
        """

        # Create empty lists of receivers.
        self.p_receivers[node_id] = set()
        self.b_receivers[node_id] = set()

        # Pick number of friends and businesses.
        n_friends = self.get_asymmetric_norm(10, 50, 200)
        n_close_friends = n_friends / 2
        n_far_friends = n_friends - n_close_friends
        n_businesses = self.get_asymmetric_norm(10, 50, 200)
        n_close_businesses = n_businesses / 2
        n_far_businesses = n_businesses - n_close_businesses

        close_businesses = set()
        far_businesses = set()
        random.shuffle(self.randomized_businesses)
        for j in range(len(self.businesses)):
            k = self.randomized_businesses[j]  # Random business ID.

            # If the business is close and we need one, we save it for i node.
            if self.close_dist(self.distance_between_nodes(node_id, k)):
                if len(close_businesses) < n_close_businesses:
                    close_businesses.add(k)
            # If the business is far and we need it, we save it for i node.
            else:
                if len(far_businesses) < n_far_businesses:
                    far_businesses.add(k)

            if len(close_businesses) >= n_close_businesses and len(far_businesses) >= n_far_businesses:
                break

        close_friends = set()
        far_friends = set()
        random.shuffle(self.randomized_non_businesses)
        for j in range(len(self.non_businesses)):
            k = self.randomized_non_businesses[j]  # Random friend ID.

            # If the business is close and we need one, we save it for i node.
            if self.close_dist(self.distance_between_nodes(node_id, k)):
                if len(close_friends) < n_close_friends:
                    close_friends.add(k)
            # If the business is far and we need it, we save it for i node.
            else:
                if len(far_friends) < n_far_friends:
                    far_friends.add(k)

            if len(close_friends) >= n_close_friends and len(far_friends) >= n_far_friends:
                break

        return(
            # We mix close and far businesses up because the payments send rate is uniform.
            # If it's not, you need to separate this and keep track of them separately.
            close_businesses.union(far_businesses),
            # We mix close and far friends up because the payments send rate is uniform.
            # If it's not, you need to separate this and keep track of them separately.
            close_friends.union(far_friends)
        )

    def generate_node_payees_bulk(self, node_ids):
        res = []
        for i in node_ids:
            res.append(self.generate_node_payees(i))
        return res

    def generate_node_bills(self, bill_ids, total, node_id):
        bills_part = {}

        # Split total into specific bills.
        node_bills = self.split_int(total, len(bill_ids))
        for i in range(len(bill_ids)):
            bill_size = node_bills[i]
            bill_id = bill_ids[i]
            # Create a bill.
            bills_part[bill_id] = {
                'size': bill_size,
                'owner': node_id,
                'cluster': random.randrange(self.CLUSTERS)
            }

        return bills_part

    def generate_bills(self):

        with Pool(self.PROCESSES) as pool:
            totals = pool.starmap(self.get_asymmetric_norm, [(self.NET_WORTH_MIN, self.NET_WORTH_AVG, self.NET_WORTH_MAX) for _ in range(self.NODES)])
            n_bills = pool.starmap(self.get_asymmetric_norm, [(self.BILLS_PP_MIN, self.BILLS_PP_AVG, self.BILLS_PP_MAX) for _ in range(self.NODES)])

        total_bills = 0
        bill_ids = []
        for i in range(self.NODES):
            # In case we have more bills than cents.
            if n_bills[i] > totals[i]:
                n_bills[i] = totals[i]

            # Generate bill IDs buckets.
            node_bill_ids = []
            for j in range(n_bills[i]):
                total_bills += 1
                node_bill_ids.append(total_bills-1)
            bill_ids.append(node_bill_ids)

        # Generate bills in parallel.
        with Pool(self.PROCESSES) as pool:
            nodes_bills = pool.starmap(self.generate_node_bills, [(bill_ids[x], totals[x], x) for x in range(self.NODES)])

        # Save generated results into global vars.
        for i in range(self.NODES):
            self.wallets[i] = list(nodes_bills[i].keys())
            self.bills.update(nodes_bills[i])

        # bill_id = -1
        # for i in range(self.NODES):
        #     # Make a blank wallet for this node.
        #     self.wallets[i] = []
        #
        #     # In cents
        #     total = self.get_asymmetric_norm(self.NET_WORTH_MIN, self.NET_WORTH_AVG, self.NET_WORTH_MAX)
        #     n_bills = self.get_asymmetric_norm(self.BILLS_PP_MIN, self.BILLS_PP_AVG, self.BILLS_PP_MAX)
        #
        #     # In case we have more bills than cents.
        #     if n_bills > total:
        #         total = n_bills
        #
        #     # Split total into specific bills.
        #     node_bills = self.split_int(total, n_bills)
        #     for bill_size in node_bills:
        #         # Create a bill.
        #         bill_id += 1
        #         self.bills[bill_id] = {
        #             'size': bill_size,
        #             'owner': i,
        #             'cluster': random.randrange(self.CLUSTERS)
        #         }
        #         # Save the bill into a wallet.
        #         self.wallets[i].append(bill_id)

    def split_int(self, num, n_pieces) -> list:
        """Splits integer num into # of random n_pieces."""
        assert num >= n_pieces >= 1

        pieces = []
        for i in range(n_pieces - 1):
            pieces.append(random.randint(1, num - (n_pieces - i) + 1))
            num -= pieces[-1]
        pieces.append(num)

        return pieces

if __name__ == '__main__':
    simulator = Simulator()



