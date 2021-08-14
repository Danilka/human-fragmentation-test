from multiprocessing import Pool, Value
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

        self.next_bill_id = Value('i', 0)

        # {owner_id} -> [bill_id, bill_id, ...]
        self.wallets = {}

        # Set of all nodes that are businesses.
        # (business_id, business_id, business_id, ...)
        self.businesses = set()
        self.randomized_businesses = list()

        # Set of all nodes that are not businesses.
        # (private_node_id, private_node_id, private_node_id, ...)
        self.non_businesses = set()
        self.randomized_non_businesses = list()

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
        return node_id // self.NODES_PER_CLUSTER

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

        # Set global counter to the next bill ID (last ID + 1).
        self.next_bill_id.value = bill_ids[-1][-1] + 1

        # Generate bills in parallel.
        with Pool(self.PROCESSES) as pool:
            nodes_bills = pool.starmap(self.generate_node_bills, [(bill_ids[x], totals[x], x) for x in range(self.NODES)])

        # Save generated results into global vars.
        for i in range(self.NODES):
            self.wallets[i] = list(nodes_bills[i].keys())
            self.bills.update(nodes_bills[i])

    @staticmethod
    def split_int(num, n_pieces) -> list:
        """Splits integer num into # of random n_pieces."""
        assert num >= n_pieces >= 1

        pieces = []
        for i in range(n_pieces - 1):
            pieces.append(random.randint(1, num - (n_pieces - i) + 1))
            num -= pieces[-1]
        pieces.append(num)

        return pieces

    @staticmethod
    def bit_distance(id1, id2):
        """Get mathematical distance between 2 integers."""
        return bin(id1 ^ id2).count("1")

    def send_amount(self, from_node_id: int, to_node_id: int, amount: float, check_balance: bool = True) -> bool:
        """
        Send an amount from one node to another.
        :param from_node_id:
        :param to_node_id:
        :param amount:
        :param check_balance: True - checks balance before sending anything. False - starts sending regardless.
        :return: True - Amount sent. False - balance is too low.
        """
        # Check if the node has enough money for the transaction.
        if check_balance:
            if self.get_balance(from_node_id) < amount:
                return False

        to_cluster_id = self.node_to_cluster(to_node_id)
        from_cluster_id = self.node_to_cluster(from_node_id)
        self.wallets[from_node_id] = sorted(
            self.wallets[from_node_id],
            key=lambda x: (
                # Sort by distance to the receiver's cluster first, excluding bills in sender's cluster.
                float("inf") if self.bills[x]['cluster'] == from_cluster_id else Simulator.bit_distance(
                    self.bills[x]['cluster'],
                    to_cluster_id,
                ),
                self.bills[x]['size'],  # Sort by the smallest bill size second.
            ),
            reverse=True,   # So we can pop the bills from the end.
        )

        amount_left_to_send = amount
        # Sending until we ren out of the needed amount or bills.
        while amount_left_to_send and len(self.wallets[from_node_id]):
            # Take the bill that we are operating with.
            bill_id = self.wallets[from_node_id].pop()

            # If the bill is not enough to cover the transaction, we send the whole bill.
            if self.bills[bill_id]['size'] < amount_left_to_send:
                amount_left_to_send -= self.bills[bill_id]['size']

                # Send the whole bill.
                self.bills[bill_id]['owner'] = to_node_id
                self.wallets[to_node_id].append(bill_id)
            # If the bill is more than we need, we split it and send a part.
            else:
                amount_left_to_send = 0.0
                new_bill_id, new_bill, bill = self.split_bill(self.bills[bill_id], amount_left_to_send)

                # Return chopped bill back to the owner.
                self.bills[bill_id] = bill
                self.wallets[from_node_id].append(bill_id)

                # Send the new bill.
                new_bill['owner'] = to_node_id
                self.bills[new_bill_id] = new_bill
                self.wallets[to_node_id].append(new_bill_id)

    def get_balance(self, node_id: int) -> float:
        """Get balance of specified node."""
        total = 0.0
        for bill_id in self.wallets[node_id]:
            total += self.bills[bill_id]['size']
        return total

    def get_next_bill_id(self):
        """Returns the next bill ID. Thread safe."""
        next_id = self.next_bill_id
        self.next_bill_id.value += 1
        return next_id

    def combine_nodes_bills(self, node_id):
        """Combines all node's bills that can be combined."""

        # Sort all node's bills by cluster_id.
        self.wallets[node_id] = sorted(self.wallets[node_id], key=lambda x: self.bills[x]['cluster'])

        # Iterate over all bills and see if any could be combined.
        i = 1
        while i < len(self.wallets[node_id]):
            bill1_id = self.wallets[node_id][i-1]
            bill2_id = self.wallets[node_id][i]

            if self.bills[bill1_id]['cluster'] == self.bills[bill2_id]['cluster']:
                # This will combine the bills and remove second id from the wallet, so no need to iterate i.
                self.combine_two_bills(bill1_id, bill2_id)
            else:
                i += 1

    def combine_two_bills(self, bill1_id, bill2_id):
        """Combines two bills into the first one. Can only be done for the same owner on the same cluster."""

        assert self.bills[bill1_id]['owner'] == self.bills[bill2_id]['owner']
        assert self.bills[bill1_id]['cluster'] == self.bills[bill2_id]['cluster']

        # Add the value from 2 to 1 bill.
        self.bills[bill1_id]['size'] += self.bills[bill2_id]['size']

        # Delete bill 2 from user's wallet.
        self.wallets[self.bills[bill2_id]['owner']].remove(bill2_id)

        # Delete bill2
        del self.bills[bill2_id]

    def split_bill(self, bill, amount_needed):
        """
        Splits a bill into two. One of the amount needed. Thread safe.
        You need to do the following with the returned values:
        1. Replace the old bill.
        2. Add new bill to self.bills
        3. Add the new bill to owner's wallet.
        :param bill: Old bill object.
        :param amount_needed: Float of the amount of the new bill (less than the old bill).
        :return: (new bill id, new bill object, updated old bill)
        """

        assert amount_needed < bill['size']

        # Lower the size of previous bill.
        bill['size'] -= amount_needed

        # Create a new bill.
        # {size: 123, owner: node_id, cluster: cluster_id}
        new_bill = {
            'size': amount_needed,
            'owner': bill['owner'],
            'cluster_id': bill['cluster'],
        }
        return (
            self.get_next_bill_id(),
            new_bill,
            bill,
        )

if __name__ == '__main__':
    simulator = Simulator()



