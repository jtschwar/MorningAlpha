// CSV parsing and data management utilities

function parseCSV(csv) {
    const lines = csv.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    // Find the return column (could be Return_3M_%, Return_6M_%, or Return_YTD_%)
    const returnCol = headers.find(h => h.startsWith('Return_'));
    const metric = returnCol ? returnCol.split('_')[1] : '3M';
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length >= 5) {
            data.push({
                Rank: parseInt(values[0]) || i,
                Ticker: values[1].trim(),
                Name: values[2].trim(),
                Exchange: values[3].trim(),
                ReturnPct: parseFloat(values[4]) || 0
            });
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
