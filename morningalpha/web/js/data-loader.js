// CSV parsing and data management utilities

function parseCSV(csv) {
    const lines = csv.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    // Find the return column (could be Return_3M_%, Return_6M_%, or Return_YTD_%)
    const returnCol = headers.find(h => h.startsWith('Return_'));
    const metric = returnCol ? returnCol.split('_')[1] : '3M';
    
    // Find column indices dynamically
    const getColIndex = (colName) => headers.findIndex(h => h === colName || h.toLowerCase() === colName.toLowerCase());
    
    const rankIdx = getColIndex('Rank');
    const tickerIdx = getColIndex('Ticker');
    const nameIdx = getColIndex('Name');
    const exchangeIdx = getColIndex('Exchange');
    const returnIdx = headers.findIndex(h => h.startsWith('Return_'));
    const sharpeIdx = getColIndex('SharpeRatio');
    const sortinoIdx = getColIndex('SortinoRatio');
    const maxDrawdownIdx = getColIndex('MaxDrawdown');
    const consistencyIdx = getColIndex('ConsistencyScore');
    const volumeTrendIdx = getColIndex('VolumeTrend');
    const qualityIdx = getColIndex('QualityScore');
    const marketCapIdx = getColIndex('MarketCap');
    const marketCapCategoryIdx = getColIndex('MarketCapCategory');
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length >= 5) {
            const rsiIdx = getColIndex('RSI');
            const momentumAccelIdx = getColIndex('MomentumAccel');
            const priceVsHighIdx = getColIndex('PriceVs20dHigh');
            const volumeSurgeIdx = getColIndex('VolumeSurge');
            const entryScoreIdx = getColIndex('EntryScore');
            
            const stock = {
                Rank: rankIdx >= 0 ? parseInt(values[rankIdx]) || i : i,
                Ticker: tickerIdx >= 0 ? values[tickerIdx].trim() : values[1].trim(),
                Name: nameIdx >= 0 ? values[nameIdx].trim() : values[2].trim(),
                Exchange: exchangeIdx >= 0 ? values[exchangeIdx].trim() : values[3].trim(),
                ReturnPct: returnIdx >= 0 ? parseFloat(values[returnIdx]) || 0 : parseFloat(values[4]) || 0,
                SharpeRatio: sharpeIdx >= 0 && values[sharpeIdx] ? parseFloat(values[sharpeIdx]) : null,     
                SortinoRatio: sortinoIdx >= 0 && values[sortinoIdx] ? parseFloat(values[sortinoIdx]) : null,    
                MaxDrawdown: maxDrawdownIdx >= 0 && values[maxDrawdownIdx] ? parseFloat(values[maxDrawdownIdx]) : null,     
                ConsistencyScore: consistencyIdx >= 0 && values[consistencyIdx] ? parseFloat(values[consistencyIdx]) : null,
                VolumeTrend: volumeTrendIdx >= 0 && values[volumeTrendIdx] ? parseFloat(values[volumeTrendIdx]) : null,     
                QualityScore: qualityIdx >= 0 && values[qualityIdx] ? parseFloat(values[qualityIdx]) : null,
                // NEW: Short-term metrics
                RSI: rsiIdx >= 0 && values[rsiIdx] ? parseFloat(values[rsiIdx]) : null,
                MomentumAccel: momentumAccelIdx >= 0 && values[momentumAccelIdx] ? parseFloat(values[momentumAccelIdx]) : null,
                PriceVs20dHigh: priceVsHighIdx >= 0 && values[priceVsHighIdx] ? parseFloat(values[priceVsHighIdx]) : null,
                VolumeSurge: volumeSurgeIdx >= 0 && values[volumeSurgeIdx] ? parseFloat(values[volumeSurgeIdx]) : null,
                EntryScore: entryScoreIdx >= 0 && values[entryScoreIdx] ? parseFloat(values[entryScoreIdx]) : null
            };
            
            // Add market cap if available
            if (marketCapIdx >= 0 && values[marketCapIdx]) {
                stock.MarketCap = parseFloat(values[marketCapIdx]) || null;
            }
            if (marketCapCategoryIdx >= 0 && values[marketCapCategoryIdx]) {
                stock.MarketCapCategory = values[marketCapCategoryIdx].trim() || null;
            }
            
            data.push(stock);
        }
    }
    
    return {
        data: data,
        metadata: {
            metric: metric,
            totalAnalyzed: data.length,
            topCount: data.length
        }
    };
}


function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            result.push(current);
            current = '';
        } else {
            current += char;
        }
    }
    result.push(current);
    
    return result;
}

function exportToJSON(data, filename = 'stock_data.json') {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    downloadBlob(blob, filename);
}

function exportToCSV(data, filename = 'stock_data.csv') {
    if (!data || data.length === 0) return;
    
    const headers = Object.keys(data[0]);
    const csv = [
        headers.join(','),
        ...data.map(row => 
            headers.map(h => {
                const val = row[h];
                return typeof val === 'string' && val.includes(',') ? `"${val}"` : val;
            }).join(',')
        )
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    downloadBlob(blob, filename);
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function filterByExchange(data, exchange) {
    return data.filter(s => s.Exchange === exchange);
}

function filterByReturnRange(data, minReturn, maxReturn) {
    return data.filter(s => s.ReturnPct >= minReturn && s.ReturnPct <= maxReturn);
}

function searchStocks(data, query) {
    const q = query.toLowerCase();
    return data.filter(s => 
        s.Ticker.toLowerCase().includes(q) ||
        s.Name.toLowerCase().includes(q)
    );
}

function calculateStats(data) {
    if (!data || data.length === 0) return null;
    
    const returns = data.map(s => s.ReturnPct).sort((a, b) => a - b);
    const sum = returns.reduce((a, b) => a + b, 0);
    const mean = sum / returns.length;
    
    const median = returns.length % 2 === 0
        ? (returns[returns.length / 2 - 1] + returns[returns.length / 2]) / 2
        : returns[Math.floor(returns.length / 2)];
    
    const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    
    return {
        count: returns.length,
        min: returns[0],
        max: returns[returns.length - 1],
        mean: mean,
        median: median,
        stdDev: stdDev,
        q1: returns[Math.floor(returns.length * 0.25)],
        q3: returns[Math.floor(returns.length * 0.75)]
    };
}
