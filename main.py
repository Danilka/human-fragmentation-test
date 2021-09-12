import copy
from multiprocessing import Pool, Value
import math
import numpy as np
import random
from timer_cm import Timer
from scipy.stats import truncnorm
from statistics import mean
import logging


log = logging.getLogger(__name__)
next_bill_id = Value('i', 0)


class Simulator:

    PROCESSES = 7

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

    def run(self, transactions: int = 1000):
        global log

        for i in range(transactions):
            with Timer("Transactions run #{}".format(i)):
                self.transactions_run()


    @classmethod
    def run_transactions_thread(cls, args: tuple) -> (dict, dict):
        """
        Run transactions from the transactions_map using passed data.
        All of the following params are passed as one tuple to simplify multiprocessing.
        :param transactions_map: List of tuples tu run transactions (from_node_id, to_node_id)
        :param wallets: Dict of wallets with node_id as a key.
        :param bills: Dict of bills from wallets with bill_id as a key.
        :return: (wallets, bills) <- New bills and wallets after the transaction.
        """
        transactions_map, wallets, bills = args
        for from_node_id, to_node_id in transactions_map:
            # Get balance.
            amount = cls.get_balance_static(from_node_id, wallets, bills) * random.uniform(0.01, 1.0) ** 2

            # Perform the transaction.
            cls.send_amount(from_node_id, to_node_id, amount, wallets, bills)

            # Merge bills.
            cls.merge_nodes_bills(to_node_id, wallets, bills)

        return wallets, bills

    def transactions_run(self):
        """Run a random set of transaction on the existing nodes in parallel."""
        # Pick the nodes for the transaction.
        # TODO: This is too uniform. You can switch it to a more sporadic approach.
        from_nodes = [x for x in range(self.NODES)]
        random.shuffle(from_nodes)

        # Generate recipients.
        # from_node_id = random.randrange(self.NODES)
        to_nodes = [self.pick_recipient(x) for x in from_nodes]

        # Prep service vars.
        node_to_bucket = {}
        # Transactions mappings (from node id, to node id)
        transactions_buckets = [[] for _ in range(self.PROCESSES)]
        # Node IDs that this bucket would interact with.
        nodes_buckets = [set() for _ in range(self.PROCESSES)]

        for i in range(self.NODES):
            from_node_id = from_nodes[i]
            from_node_bucket = node_to_bucket[from_node_id] if from_node_id in node_to_bucket.keys() else None
            to_node_id = to_nodes[i]
            to_node_bucket = node_to_bucket[to_node_id] if to_node_id in node_to_bucket.keys() else None

            if from_node_bucket is None and to_node_bucket is None:
                from_node_bucket = i % self.PROCESSES
                to_node_bucket = from_node_bucket
            elif from_node_bucket and to_node_bucket is None:
                to_node_bucket = from_node_bucket
            elif from_node_bucket is None and to_node_bucket:
                from_node_bucket = to_node_bucket
            elif from_node_bucket == to_node_bucket:
                # All the buckets are already assigned properly.
                pass
            else:
                # Nodes are already in different buckets, the transaction is a no go.
                continue

            # Save the bucket mapping.
            node_to_bucket[from_node_id] = from_node_bucket
            node_to_bucket[to_node_id] = to_node_bucket

            # At this point from_node_bucket = to_node_bucket
            transactions_buckets[from_node_bucket].append(
                (from_node_id, to_node_id)
            )

            # Save the IDs into the mapping.
            nodes_buckets[from_node_bucket].update([from_node_id, to_node_id])

        # Split the data and run transactions.
        # List with process id as a pointer to the same structure as self.wallets
        wallets_split = [{} for _ in range(self.PROCESSES)]
        # List with process id as a pointer to the same structure as self.bills
        bills_split = [{} for _ in range(self.PROCESSES)]

        for node_id, bucket_id in node_to_bucket.items():
            wallets_split[bucket_id][node_id] = self.wallets[node_id]
            for bill_id in wallets_split[bucket_id][node_id]:
                bills_split[bucket_id][bill_id] = self.bills[bill_id]

        # Run transactions in parallel.
        # Arguments prep.
        parallel_args = []
        for i in range(self.PROCESSES):
            parallel_args.append(
                (transactions_buckets[i], wallets_split[i], bills_split[i])
            )
        # Run transactions. Choose the method below and comment one out.
        # In parallel:
        with Pool(self.PROCESSES) as pool:
            parallel_res = pool.map(self.run_transactions_thread, parallel_args)
        # # In one thread sequentially:
        # parallel_res = []
        # for i in range(len(parallel_args)):
        #     parallel_res.append(
        #         self.run_transactions_thread(parallel_args[i])
        #     )

        # Merge the data back to the main pull.
        # Re-save the bills of the nodes that did not participate in this run.
        new_bills = {}
        not_participating_nodes = set([x for x in range(self.NODES)]) - set(node_to_bucket.keys())
        for node_id in not_participating_nodes:
            for bill_id in self.wallets[node_id]:
                new_bills[bill_id] = self.bills[bill_id]

        for node_id, bucket_id in node_to_bucket.items():
            # Update the wallets.
            self.wallets[node_id] = parallel_res[bucket_id][0][node_id]     # [0] is wallets

            # Update bills from these wallets.
            for bill_id in self.wallets[node_id]:
                new_bills[bill_id] = parallel_res[bucket_id][1][bill_id]    # [1] is bills

        # Flush bills into the common pull.
        self.bills = new_bills

    def pick_recipient(self, node_id):
        """Pick a recipient for a transaction from node_id."""
        if random.uniform(0, 1) > 0.8:
            # Transaction to a private person.
            return random.choice(list(self.p_receivers[node_id]))
        else:
            # Transaction to a business.
            return random.choice(list(self.b_receivers[node_id]))

    def system_status(self):
        global log

        log.info("---------System Status---------")
        log.info("{} nodes.".format(len(self.nodes_loc)))
        log.info("Mean friends per person: {}".format(mean([len(self.p_receivers[i]) for i in range(self.NODES)])))
        log.info("Mean businesses per person: {}".format(mean([len(self.b_receivers[i]) for i in range(self.NODES)])))

        totals = [self.bills[x]['size'] for x in self.bills.keys()]
        log.info("{}$ in the system.".format(sum(totals)/100))
        log.info("Mean bill size ${}".format(mean(totals)/100))
        log.info("Total bills: {}".format(len(self.bills.keys())))
        log.info("Avg bills per person: {}".format(len(self.bills.keys())/self.NODES))

        wealth = [sum([self.bills[x]['size'] for x in self.wallets[i]]) for i in range(self.NODES)]
        log.info("Mean wealth per person ${}".format(mean(wealth)/100))
        log.info("Max wealth: ${}".format(max(wealth)/100))
        log.info("Min wealth: ${}".format(min(wealth)/100))
        log.info("-------------------------------")

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

    @classmethod
    def node_to_cluster(cls, node_id: int) -> int:
        """Get cluster_id that node_id belongs to."""
        return node_id // cls.NODES_PER_CLUSTER

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
            bill_size = float(node_bills[i])
            bill_id = bill_ids[i]
            # Create a bill.
            bills_part[bill_id] = {
                'size': bill_size,
                'owner': node_id,
                'cluster': random.randrange(self.CLUSTERS)
            }

        return bills_part

    def generate_bills(self):
        global next_bill_id

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
        next_bill_id.value = bill_ids[-1][-1] + 1

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

    @classmethod
    def send_amount(cls, from_node_id: int, to_node_id: int, amount: float, wallets: dict, bills: dict) -> bool:
        """
        Send an amount from one node to another.
        :param from_node_id:
        :param to_node_id:
        :param amount:
        :param wallets: Gets updated by a reference.
        :param bills: Gets updated by a reference.
        :return: True - Amount sent. False - balance is too low.
        """
        # We assume that the balance is correct and we do not need to check it.
        # if check_balance:
        #     if self.get_balance(from_node_id) < amount:
        #         return False

        to_cluster_id = cls.node_to_cluster(to_node_id)
        from_cluster_id = cls.node_to_cluster(from_node_id)
        wallets[from_node_id] = sorted(
            wallets[from_node_id],
            key=lambda x: (
                # Sort by distance to the receiver's cluster first, excluding bills in sender's cluster.
                float("inf") if bills[x]['cluster'] == from_cluster_id else Simulator.bit_distance(
                    bills[x]['cluster'],
                    to_cluster_id,
                ),
                bills[x]['size'],  # Sort by the smallest bill size second.
            ),
            reverse=True,   # So we can pop the bills from the end.
        )

        amount_left_to_send = amount
        # Sending until we ren out of the needed amount or bills.
        while amount_left_to_send and len(wallets[from_node_id]):
            # Take the bill that we are operating with.
            bill_id = wallets[from_node_id].pop()

            # If the bill is not enough to cover the transaction, we send the whole bill.
            if bills[bill_id]['size'] <= amount_left_to_send:
                # Send the whole bill.
                bills[bill_id]['owner'] = to_node_id
                wallets[to_node_id].append(bill_id)

                amount_left_to_send -= bills[bill_id]['size']
            # If the bill is more than we need, we split it and send a part.
            else:
                new_bill_id, new_bill, bill = cls.split_bill(bills[bill_id], amount_left_to_send)

                # Return chopped bill back to the owner.
                bills[bill_id] = bill
                wallets[from_node_id].append(bill_id)

                # Send the new bill.
                new_bill['owner'] = to_node_id
                bills[new_bill_id] = new_bill
                wallets[to_node_id].append(new_bill_id)

                amount_left_to_send = 0.0

                cls.check_wallet(wallets, bills, from_node_id, "Error in the senders wallet.")
                cls.check_wallet(wallets, bills, to_node_id, "Error in the receiver's wallet.")

    @staticmethod
    def check_wallet(wallets, bills, owner_id, msg=""):
        """
        Checks if all the bills in the wallet belong to their owner.
        :param wallets:
        :param bills:
        :param owner_id: Node ID
        :param msg: Optional message before the error output.
        :return:
        """
        global log

        for bill_id in wallets[owner_id]:
            if bills[bill_id]['owner'] != owner_id:
                if msg:
                    log.error(msg)
                log.error(
                    "Bill #{} is in the wallet of Node ID #{}, but has the owner set to #{}. Bill: {}".format(
                        bill_id,
                        owner_id,
                        bills[bill_id]['owner'],
                        bills[bill_id],
                    )
                )

    @staticmethod
    def get_balance_static(node_id: int, wallets: dict, bills: dict) -> float:
        """Get balance of a specified node using passed data."""
        total = 0.0
        for bill_id in wallets[node_id]:
            total += bills[bill_id]['size']
        return total

    def get_balance(self, node_id: int) -> float:
        """Get balance of a specified node."""
        return self.get_balance_static(node_id, self.wallets, self.bills)

    @staticmethod
    def get_next_bill_id():
        """Returns the next bill ID. Thread safe."""
        global next_bill_id

        next_id = next_bill_id.value
        next_bill_id.value += 1
        return next_id

    @classmethod
    def merge_nodes_bills(cls, node_id: int, wallets: dict, bills: dict):
        """
        Combines all node's bills that can be combined.
        :param node_id: Node ID for which we are trying to merge the bills.
        :param wallets: Gets updated by reference.
        :param bills: Gets updated by reference.
        :return:
        """

        # Sort all node's bills by cluster_id.
        wallets[node_id] = sorted(wallets[node_id], key=lambda x: bills[x]['cluster'])

        # Iterate over all bills and see if any could be combined.
        i = 1
        while i < len(wallets[node_id]):
            bill1_id = wallets[node_id][i-1]
            bill2_id = wallets[node_id][i]

            if bills[bill1_id]['cluster'] == bills[bill2_id]['cluster']:
                # This will combine the bills and remove second id from the wallet, so no need to iterate i.
                cls.combine_two_bills(bill1_id, bill2_id, wallets, bills)
            else:
                i += 1

    @staticmethod
    def combine_two_bills(bill1_id, bill2_id, wallets, bills):
        """
        Combines two bills into the first one. Can only be done for the same owner on the same cluster.
        :param bill1_id: Bill ID of the first bill to merge.
        :param bill2_id: Bill ID of the second bill to merge.
        :param wallets: Gets updated by reference.
        :param bills: Gets updated by reference.
        :return:
        """
        global log

        # assert bills[bill1_id]['owner'] == bills[bill2_id]['owner']
        if bills[bill1_id]['owner'] != bills[bill2_id]['owner']:
            # log.warning("Trying to combine bills by different owners.")
            # log.warning("Bill 1: {}".format(bills[bill1_id]))
            # log.warning("Bill 2: {}".format(bills[bill2_id]))
            return
        # assert bills[bill1_id]['cluster'] == bills[bill2_id]['cluster']
        if bills[bill1_id]['cluster'] != bills[bill2_id]['cluster']:
            # log.warning("Trying to combine bills that are not in the same cluster.")
            # log.warning("Bill 1: {}".format(bills[bill1_id]))
            # log.warning("Bill 2: {}".format(bills[bill2_id]))
            return

        # Add the value from 2 to 1 bill.
        bills[bill1_id]['size'] += bills[bill2_id]['size']

        # Delete bill 2 from user's wallet.
        wallets[bills[bill2_id]['owner']].remove(bill2_id)

        # Delete bill2
        del bills[bill2_id]

    @classmethod
    def split_bill(cls, bill, amount_needed):
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
        bill['size'] -= float(amount_needed)

        # Create a new bill.
        # {size: 123, owner: node_id, cluster: cluster_id}
        new_bill = {
            'size': amount_needed,
            'owner': bill['owner'],
            'cluster': bill['cluster'],
        }
        return (
            cls.get_next_bill_id(),
            new_bill,
            bill,
        )


if __name__ == '__main__':
    simulator = Simulator()
    simulator.run(10)
    simulator.system_status()

