from DailyStocks import Equities, Portfolio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_eff_frontier(universe):

    start = round(max(universe.min_ind_annual_return(), 0), 1)
    end = round(universe.max_ind_annual_return(), 1)
    returns = np.arange(start, end, 0.1)

    return_vec = []
    risk_vec = []
    exp_returns = universe.daily_returns()
    cov_mtrx = universe.cov_matrix()
    for target in returns:
        holdings = universe.mark_portfolio(target)
        markowitz = Portfolio(holdings, exp_returns, cov_mtrx)

        return_vec.append(markowitz.port_return())
        risk_vec.append(markowitz.port_risk())

    output = {'returns': return_vec, 'risk': risk_vec}

    return output


def plot_eqlwt_vs_frontier(universe):

    # Get Efficient Markowitz Frontier
    frontier = get_eff_frontier(universe)

    # Get Equal Weighted Portfolio for Comparison
    holdings = np.zeros(len(universe.stocks))
    holdings = pd.DataFrame(holdings + 1/len(universe.stocks))
    holdings.index = sorted(universe.stocks)
    current_port = Portfolio(holdings, universe.daily_returns(), universe.cov_matrix())

    # Plot Efficient Frontier against equally weighted Portfolio
    plt.scatter(frontier['risk'], frontier['returns'], c='b', label='Efficient Frontier')
    plt.scatter(current_port.port_risk(), current_port.port_return(), c='r', label='Equal Weighted Portfolio')
    plt.legend(loc='upper left')
    plt.xlim = (0, max(frontier['risk']))
    plt.ylim = (0, max(frontier['returns']))
    plt.title('Markowitz Portfolio')
    plt.show()
    return


if __name__ == '__main__':
    universe = Equities(['AAPL', 'MSFT', 'TSLA', 'GS', 'BA', 'VZ', 'XOM',
                         'MMM', 'GE', 'WMT'], '2021-1-1', '2021-4-1')

    plot_eqlwt_vs_frontier(universe)


