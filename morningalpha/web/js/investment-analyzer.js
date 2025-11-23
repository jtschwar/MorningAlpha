// Investment analysis and scoring functions

/**
 * Calculate investment score (0-100) based on multiple factors
 * Higher score = better investment opportunity
 */
function calculateInvestmentScore(stock) {
    if (!stock || stock.ReturnPct == null) return null;
    
    let score = 0;
    let factors = 0;
    
    // Return component (30% weight)
    if (stock.ReturnPct != null) {
        // Normalize return: 0-200% maps to 0-30 points
        const returnScore = Math.min(30, (stock.ReturnPct / 200) * 30);
        score += returnScore;
        factors += 30;
    }
    
    // Quality score component (25% weight)
    if (stock.QualityScore != null && !isNaN(stock.QualityScore)) {
        score += (stock.QualityScore / 100) * 25;
        factors += 25;
    }
    
    // Sharpe ratio component (20% weight)
    if (stock.SharpeRatio != null && !isNaN(stock.SharpeRatio)) {
        // Normalize Sharpe: -1 to 3 maps to 0-20 points
        const sharpeScore = Math.max(0, Math.min(20, ((stock.SharpeRatio + 1) / 4) * 20));
        score += sharpeScore;
        factors += 20;
    }
    
    // Consistency component (15% weight)
    if (stock.ConsistencyScore != null && !isNaN(stock.ConsistencyScore)) {
        score += (stock.ConsistencyScore / 100) * 15;
        factors += 15;
    }
    
    // Drawdown penalty (10% weight) - penalize high drawdowns
    if (stock.MaxDrawdown != null && !isNaN(stock.MaxDrawdown)) {
        // Less negative drawdown = better (closer to 0)
        // -50% drawdown = 0 points, 0% drawdown = 10 points
        const drawdownScore = Math.max(0, Math.min(10, ((stock.MaxDrawdown + 50) / 50) * 10));
        score += drawdownScore;
        factors += 10;
    }
    
    // Normalize to 0-100 scale if we have all factors
    if (factors > 0) {
        score = (score / factors) * 100;
    }
    
    return Math.round(score * 10) / 10; // Round to 1 decimal
}

/**
 * Calculate risk level based on volatility and drawdown
 */
function calculateRiskLevel(stock) {
    if (!stock) return 'unknown';
    
    const drawdown = stock.MaxDrawdown || 0;
    const sharpe = stock.SharpeRatio || 0;
    
    // Risk assessment based on drawdown and Sharpe
    if (drawdown < -40 || sharpe < -1) {
        return 'very-high';
    } else if (drawdown < -30 || sharpe < 0) {
        return 'high';
    } else if (drawdown < -20 || sharpe < 0.5) {
        return 'moderate';
    } else if (drawdown < -10 && sharpe > 1) {
        return 'low';
    } else {
        return 'moderate';
    }
}

/**
 * Calculate risk/reward ratio
 * Higher ratio = better risk-adjusted opportunity
 */
function calculateRiskRewardRatio(stock) {
    if (!stock || stock.ReturnPct == null) return null;
    
    const returnPct = Math.abs(stock.ReturnPct);
    const drawdown = Math.abs(stock.MaxDrawdown || 1);
    
    if (drawdown === 0) return returnPct;
    
    return returnPct / drawdown;
}

/**
 * Apply investment filters based on user preferences
 */
function applyInvestmentFilters(data) {
    const riskTolerance = document.getElementById('riskTolerance').value;
    const minQuality = parseInt(document.getElementById('minQuality').value);
    const maxDrawdown = parseFloat(document.getElementById('maxDrawdown').value);
    
    let filtered = [...data];
    
    // Filter by quality score
    if (minQuality > 0) {
        filtered = filtered.filter(s => 
            s.QualityScore != null && s.QualityScore >= minQuality
        );
    }
    
    // Filter by max drawdown
    if (maxDrawdown > -100) {
        filtered = filtered.filter(s => 
            s.MaxDrawdown == null || s.MaxDrawdown >= maxDrawdown
        );
    }
    
    // Filter by risk tolerance
    filtered = filtered.filter(s => {
        const riskLevel = calculateRiskLevel(s);
        
        switch(riskTolerance) {
            case 'conservative':
                return riskLevel === 'low' || riskLevel === 'moderate';
            case 'moderate':
                return riskLevel !== 'very-high';
            case 'aggressive':
                return true; // Accept all risk levels
            default:
                return true;
        }
    });
    
    return filtered;
}

/**
 * Sort stocks by various criteria
 */
function sortStocks(data, sortBy) {
    const sorted = [...data];
    
    switch(sortBy) {
        case 'investmentScore':
            sorted.sort((a, b) => {
                const scoreA = a.investmentScore || 0;
                const scoreB = b.investmentScore || 0;
                return scoreB - scoreA;
            });
            break;
            
        case 'return':
            sorted.sort((a, b) => (b.ReturnPct || 0) - (a.ReturnPct || 0));
            break;
            
        case 'quality':
            sorted.sort((a, b) => {
                const qA = a.QualityScore || 0;
                const qB = b.QualityScore || 0;
                return qB - qA;
            });
            break;
            
        case 'sharpe':
            sorted.sort((a, b) => {
                const sA = a.SharpeRatio || -999;
                const sB = b.SharpeRatio || -999;
                return sB - sA;
            });
            break;
            
        case 'riskReward':
            sorted.sort((a, b) => {
                const rrA = a.riskRewardRatio || 0;
                const rrB = b.riskRewardRatio || 0;
                return rrB - rrA;
            });
            break;
            
        case 'marketCap':
            sorted.sort((a, b) => {
                const mcA = a.MarketCap || 0;
                const mcB = b.MarketCap || 0;
                return mcB - mcA; // Largest first
            });
            break;
            
        case 'entryScore':
            sorted.sort((a, b) => {
                const esA = a.EntryScore || 0;
                const esB = b.EntryScore || 0;
                return esB - esA; // Higher entry score = better
            });
            break;
            
        case 'momentumAccel':
            sorted.sort((a, b) => {
                const maA = a.MomentumAccel || 0;
                const maB = b.MomentumAccel || 0;
                return maB - maA; // Higher acceleration = better
            });
            break;
            
        default:
            // Default: investment score
            sorted.sort((a, b) => {
                const scoreA = a.investmentScore || 0;
                const scoreB = b.investmentScore || 0;
                return scoreB - scoreA;
            });
    }
    
    return sorted;
}

/**
 * Calculate portfolio-level metrics
 */
function calculatePortfolioMetrics(portfolio) {
    if (!portfolio || portfolio.length === 0) {
        return {
            avgReturn: 0,
            avgQuality: 0,
            avgSharpe: 0,
            worstDrawdown: 0,
            riskDistribution: 'N/A'
        };
    }
    
    const returns = portfolio.map(s => s.ReturnPct || 0);
    const qualities = portfolio.map(s => s.QualityScore || 0).filter(q => q > 0);
    const sharpes = portfolio.map(s => s.SharpeRatio || 0).filter(s => !isNaN(s) && s !== 0);
    const drawdowns = portfolio.map(s => s.MaxDrawdown || 0);
    
    const riskLevels = portfolio.map(s => calculateRiskLevel(s));
    const riskCounts = {
        'low': riskLevels.filter(r => r === 'low').length,
        'moderate': riskLevels.filter(r => r === 'moderate').length,
        'high': riskLevels.filter(r => r === 'high').length,
        'very-high': riskLevels.filter(r => r === 'very-high').length
    };
    
    const riskDistribution = Object.entries(riskCounts)
        .filter(([_, count]) => count > 0)
        .map(([level, count]) => `${level}: ${count}`)
        .join(', ');
    
    return {
        avgReturn: returns.reduce((a, b) => a + b, 0) / returns.length,
        avgQuality: qualities.length > 0 
            ? qualities.reduce((a, b) => a + b, 0) / qualities.length 
            : 0,
        avgSharpe: sharpes.length > 0 
            ? sharpes.reduce((a, b) => a + b, 0) / sharpes.length 
            : 0,
        worstDrawdown: Math.min(...drawdowns),
        riskDistribution: riskDistribution || 'N/A'
    };
}

