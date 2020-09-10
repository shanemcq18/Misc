# markov_chains.py
"""Markov models, with applications to natural language processing.

MarkovChain: Base Markov chain class with transitioning methods.
SentenceGenerator: Makes a Markov chain out of a source file.

Author: Shane McQuarrie
"""

import os
import numpy as np
from scipy import sparse
from scipy import linalg as la


# TODO: use nltk to parse documents in the SentenceGenerator classes.
# TODO: put installation instructions for nltk in README.md.
# TODO: convert to package.


class MarkovChain: # ==========================================================
    """A finite, temporally homogeneous, first-order Markov Chain.

    Attributes:
        chain ((n, n) ndarray): The column-stochastic transition matrix.
        states (list(str)): The n labels that correspond to the states.
        _sparse (bool): If True, the transition matrix is stored as a
            compressed sparse column matrix (scipy.sparse.csc_matrix).
            If False, the transition matrix is stored as a NumPy array.
    """
    def __init__(self, chain, states, _sparse=False):
        """Set instance variables and check dimensions."""

        # Validate the transition matrix.
        m,n = chain.shape
        if _sparse:
            chain = sparse.csc_matrix(chain)
            col_sums = np.array(chain.sum(axis=0))[0]
        else:
            col_sums = chain.sum(axis=0)

        if not np.allclose(np.ones(n), col_sums):
            raise ValueError("Transition matrix columns do not all sum to 1")

        # Validate the number of states.
        S = len(states)
        if S != m:
            raise ValueError("{} state labels required (not {})".format(m, S))

        # Save the attributes.
        self.chain = chain
        self.states = states
        self._sparse = _sparse

    def _transition(self, j):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the jth state.
        """
        c = self.chain[:,j].toarray()[:,0] if self._sparse else self.chain[:,j]
        return np.random.multinomial(1, c).argmax()

    def walk(self, start, n_steps, labels=True):
        """Make a sequence of n_steps transitions.

        Parameters:
            start (int or str): the first state, either by index or by label.
            n_steps (int): the number of state transitions to make.
            labels (bool): if True, return the transition sequence by their
                state labels. Otherwise, return the sequence by state index.

        Returns:
            A sequence of n_step+1 state labels (labels=True) or state indices.

        """
        # Translate 'start' into an index if given as a string.
        if isinstance(start, str):
            start = self.states.index(start)

        path = [start]
        for _ in range(n_steps):
            path.append(self._transition(path[-1]))

        return [self.states[i] for i in path] if labels else path

    def search(self, start=0, stop=-1, labels=True):
        """Make a sequence of transitions from a specified start state until
        reaching a specified stop state.

        Parameters:
            start (int or str): the first state, either by index or by label.
            stop (int or str): the final state, either by index or by label.
            labels (bool): if True, return the transition sequence by their
                state labels. Otherwise, return the sequence by state index.

        Returns:
            A sequence of state labels (labels=True) or state indices (else).
        """
        # Translate 'start' into an index if given as a string.
        if isinstance(start, str):
            start = self.states.index(start)

        # Translate 'stop' into an index if given as a string.
        if isinstance(stop, str):
            stop = self.states.index(stop)
        if stop < 0:
            stop += self.chain.shape[0]

        path = [start]
        while path[-1] != stop:
            path.append(self._transition(path[-1]))

        return [self.states[i] for i in path] if labels else path

    def steady_state(self, x0=None, tol=1e-12, maxiters=100):
        """Compute a steady state distribution of the Markov chain.

        Inputs:
            x0 ((num_states,) ndarray): an (optional) initial distribution.
            tol (float):
            maxiters (int):

        Raises:
            ValueError: If the iteration does not converge within tol after
                maxiters iterations.

        Returns:

        """
        # Generate a random initial state distribution vector.
        if x0 is None:
            x0 = np.random.random(self.chain.shape[0])
            x0 /= x0.sum()

        # Run the iteration until convergence.
        for i in range(maxiters):
            x1 = self.chain.dot(x0)
            if la.norm(x0 - x1) < tol:
                return x1
            x0 = x1

        # Raise an exception after N iterations without convergence.
        raise ValueError("Iteration did not converge")

    def __str__(self):
        """String representation: the transition matrix."""
        title = "sparse Markov chain" if self._sparse else "Markov chain"
        return "{}-state {}".format(title, self.chain.shape[0])

    def __repr__(self):
        out = str(self)
        out += "\nStates:\n\t{}".format("\n\t".join(self.states))
        out += "\nTransition Matrix:\n{}".format(self.chain)
        return out

    # End of MarkovChain Class ================================================


# TODO: sparsify
class MarkovChainK(MarkovChain): # ===========================================
    """A finite, temporally homogeneous Markov Chain of order k.

    Attributes:
        chain ((m, n) ndarray): The column-stochastic transition matrix.
        states (list(str)): The m labels that correspond to the single states.
        products (list(list(int))): The n cartesian k-products of state
            indices. For example, if there are 2 states and the order is 2,
            this should be a sublist of [[0,0], [0,1], [1,0], [1,1]].
        order (int): The order of the Markov chain. In a Markov chain of order
            k, the next state depends on the previous k states.
        _sparse (bool): If True, the transition matrix is stored as a
            compressed sparse column matrix (scipy.sparse.csc_matrix).
            If False, the transition matrix is stored as a NumPy array.
    """
    def __init__(self, chain, states, products, order=1, _sparse=False):
        """Set instance variables and check dimensions."""
        MarkovChain.__init__(self, chain, states, _sparse)

        # Validate the number of product states.
        m,n = self.chain.shape
        P = len(products)
        if P != n:
            raise ValueError("{} product labels required (not {})".format(n,P))

        self.products = products
        self.order = order

    def walk(self, start, n_steps, labels=True):
        """Make a sequence of n_steps transitions.

        Parameters:
            start (list(int or str)): a sequence of k states, either by index
                or by label, where k is the order of the chain.
            n_steps (int): the number of state transitions to make.
            labels (bool): if True, return the transition sequence by their
                state labels. Otherwise, return the sequence by state index.

        Returns:
            A sequence of n_step+1 state labels (labels=True) or state indices.
        """
        # Validate the start sequence.
        if len(start) != self.order:
            raise ValueError("Start sequence must have {} states".format(
                                                                self.order))
        if isinstance(start[0], str):
            start = [self.states.index(s) for s in start]

        path = list(start)
        for _ in range(n_steps):
            current_state = self.products.index(path[-self.order:])
            path.append(self._transition(current_state))

        return [self.states[i] for i in path] if labels else path

    def search(self, start, stop=-1, labels=True):
        """Make a sequence of transitions from a specified start state until
        reaching a specified stop state.

        Parameters:
            start (list(int or str)): a sequence of k states, either by index
                or by label, where k is the order of the chain.
            stop (int or str): the final state, either by index or by label.
            labels (bool): if true, return the transition sequence by their
                state labels. Otherwise, return the sequence by state index.

        Returns:
            A single string of the state labels, or a series of state indices.

        Notes:
            May cycle indefinitely if the chain is not strongly connected.
        """
        # Validate the start sequence.
        if len(start) != self.order:
            raise ValueError("Start sequence must have {} states".format(
                                                                self.order))
        if isinstance(start[0], str):
            start = [self.states.index(s) for s in start]

        # Translate 'stop' into an index if given as a string.
        if isinstance(stop, str):
            try:
                stop = self.states.index(stop)
            except ValueError as e:
                raise ValueError("{} is not a state label".format(repr(stop)))
        if stop < 0:
            stop += self.chain.shape[0]

        path = list(start)
        while path[-1] != stop:
            current_state = self.products.index(path[-self.order:])
            path.append(self._transition(current_state))

        return [self.states[i] for i in path] if labels else path

    def steady_state(self, *args, **kwargs):
        """Compute a steady state distribution of the Markov chain.

        Inputs:
            x0 ((num_states,) ndarray): an (optional) initial distribution.
        """
        if self.order != 1:
            raise NotImplementedError("Steady states can only be computed for "
                                                    "Markov chains of order 1")
        else:
            return MarkovChain.steady_state(self, *args, **kwargs)

    def __str__(self):
        """String representation: the transition matrix."""
        title = "sparse Markov chain" if self._sparse else "Markov chain"
        return "{}-state {} of order {}".format(title, self.chain.shape[0],
                                                                    self.order)

    # End of MarkovChainK Class ===============================================


# TODO: use nltk
class SentenceGenerator(MarkovChainK): # =====================================
    """Markov chain of higher order creator for simulating English."""

    def __init__(self, filename, order=1, _sparse=False):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """

        # Build the list of states and product states.
        states = ["$tart"]
        products = []
        starts = []

        with open(filename, 'r') as source:
            for line in source:
                sentence = line.split()

                # Add to states.
                for word in sentence:
                    if word not in states:
                        states.append(word)
                sentence = ["$tart"] + sentence + ["$top"]

                # Add to start and product states if there are enough words.
                n = len(sentence)
                if n < order:
                    continue
                sequences = [sentence[i:i+order] for i in range(n-order+1)]
                if sequences[0] not in starts:
                    starts.append(sequences[0])
                for sequence in sequences:
                    if sequence not in products:
                        products.append(sequence)

        states.append("$top")
        starts = [[states.index(i) for i in start] for start in starts]
        products = [[states.index(i) for i in prod] for prod in products]

        # Initialize empty transition matrices of the appropriate size.
        self._start_probs = np.zeros(len(starts))
        if _sparse:
            self.chain = sparse.lil_matrix((len(states), len(products)))
        else:
            self.chain = np.zeros((len(states), len(products)))

        # Process the data. This assumes one sentence per line in the file.
        with open(filename, 'r') as source:
            for line in source:
                sentence = ["$tart"] + line.split() + ["$top"]
                n = len(sentence)
                if n < order:
                    continue

                # Add 1's to the appropriate parts of the transition matrix.
                indices = [states.index(word) for word in sentence]
                seqs = [sentence[i:i+order] for i in range(n-order+1)]
                seqs = [[states.index(i) for i in seq] for seq in seqs]
                for i in range(n-order):
                    self.chain[seqs[i+1][-1], products.index(seqs[i])] += 1

                # Account for start and end probabilities.
                self._start_probs[starts.index(seqs[0])] += 1
                self.chain[0, products.index(seqs[-1])] += 1

        # Make each column sum to 1.
        self._start_probs /= self._start_probs.sum(axis=0)
        if _sparse:
            self.chain = self.chain.tocsc()
            for j in range(self.chain.shape[1]):
                self.chain[:,j] /= self.chain[:,j].sum()
        else:
            self.chain /= self.chain.sum(axis=0)

        # Store the remaining attributes.
        self.filename = filename
        self.states = states
        self._start_states = starts
        self.products = products
        self.order = order
        self._sparse = _sparse

    def babble(self):
        """Write a nonsense English sentence using the Markov chain."""

        # Choose a sequence to begin with.
        start = np.random.multinomial(1, self._start_probs).argmax()
        start = self._start_states[start]

        # Transition through the chain.
        return " ".join(self.search(start, -1, labels=True)[1:-1])

    def write_file(self, filename, num_sentences=50):
        """Write num_sentences random sentences to the specified file."""

        # Check before overwriting an existing file.
        if os.path.isfile(filename):
            if raw_input("Overwrite {} [y/n]? ".format(filename)) != "y":
                print("exiting...")
                return

        # Write to the file.
        with open(filename, 'w') as out:
            for _ in range(num_sentences):
                out.write(self.babble() + '\n')

    # End of SentenceGenerator Class ==========================================
