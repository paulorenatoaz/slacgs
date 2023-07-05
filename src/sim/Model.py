class Model:
    """
    Represents a Bayes Classifier Loss Analysis Model composed by:

    • A set of Dataset cardinalities N = {n0...ni} , ni ∈ {2*k | k ∈ int*}, i.e., the cardinality on each Dataset to be analysed.
    • Each feature discrimination power, either alone or in the presence of the others features.
    • The problem dimensionality "dim", i.e., the number of available features.
    • Dictionary "dictionary", from which we will pick our classifier

    """

    def __init__(self, params: [], max_n=1024, N=[], dictionary=['linear']):

        """
        :param max_n :  last Dataset cardinality, assuming N = [2,4,6,...,max_n]
        :param params : list containning Sigma's and Rho's
        :param N : set of Dataset cardinalities N = [n0...ni]
        :param dictionary: A dictionary (also known as search space bias) is a family of classifiers (e.g., linear classifiers, quadratic classifiers,...)
        """

        dim = 2
        param_len = 3
        while param_len < len(params):
            dim += 1
            param_len += dim

        if param_len > len(params):
            raise ValueError('Check parameters list lenght')

        self.dim = dim
        self.sigma = params[0:dim]
        self.rho = params[dim:len(params)]
        self.mean_pos = [1 for d in range(dim)]
        self.mean_neg = [-1 for d in range(dim)]

        self.dictionary = dictionary

        sum = 0
        aux1 = []
        aux1.append(sum)
        for i in range(1, len(self.sigma) - 1):
            sum += len(self.sigma) - i
            aux1.append(sum)

        sum = len(self.sigma) - 1
        aux2 = []
        aux2.append(sum)
        for i in range(1, len(self.sigma) - 1):
            sum += len(self.sigma) - (i + 1)
            aux2.append(sum)

        self.rho_matrix = [[None] * (i + 1) + self.rho[aux1[i]:aux2[i]] for i in range(len(self.sigma) - 1)]

        self.cov = [[self.sigma[p] ** 2 if p == q else self.sigma[p] * self.sigma[q] * self.rho_matrix[p][
            q] if q > p else self.sigma[p] * self.sigma[q] * self.rho_matrix[q][p] for q in range(len(self.sigma))] for
                    p in range(len(self.sigma))]
        self.params = params
        self.N = N if N else list(range(2, max_n + 1, 2))
        self.max_n = max_n if not N else N[len(N) - 1]



