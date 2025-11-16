// Chart creation utilities

function createChart(type, data, metadata) {
    const container = document.getElementById('chartPlot');
    
    switch(type) {
        case 'bar':
            createBarChart(data, metadata, container);
            break;
        case 'scatter':
            createScatterPlot(data, metadata, container);
            break;
        case 'treemap':
            createTreemap(data, metadata, container);
            break;
        case 'sunburst':
            createSunburst(data, metadata, container);
            break;
        default:
            createBarChart(data, metadata, container);
    }
}

function createBarChart(data, metadata, container) {
    const reversedData = [...data].reverse();
    
    const trace = {
        x: reversedData.map(s => s.ReturnPct),
        y: reversedData.map(s => s.Ticker),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: reversedData.map(s => s.ReturnPct),
            colorscale: [
                [0, '#48bb78'],
                [0.5, '#4299e1'],
                [1, '#9f7aea']
            ],
            line: {
                color: 'rgba(255,255,255,0.5)',
                width: 1
            }
        },
        text: reversedData.map(s => `${s.ReturnPct.toFixed(2)}%`),
        textposition: 'outside',
        hovertemplate: '<b>%{y}</b><br>' +
                      'Return: %{x:.2f}%<br>' +
                      '<extra></extra>',
        customdata: reversedData.map(s => s.Ticker)
    };
    
    const layout = {
        title: {
            text: `Top ${data.length} Stock Gainers - ${metadata?.metric || ''}`,
            font: { size: 20, color: '#2d3748' }
        },
        xaxis: {
            title: 'Return (%)',
            gridcolor: '#e2e8f0'
        },
        yaxis: {
            title: 'Ticker',
            automargin: true
        },
        height: Math.max(500, data.length * 25),
        margin: { l: 100, r: 100, t: 60, b: 60 },
        plot_bgcolor: '#f7fafc',
        paper_bgcolor: 'white'
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true});
    
    container.on('plotly_click', function(data) {
        const ticker = data.points[0].customdata;
        viewStockDetail(ticker);
    });
}

function createScatterPlot(data, metadata, container) {
    const exchanges = [...new Set(data.map(s => s.Exchange))];
    const traces = exchanges.map(exchange => {
        const filtered = data.filter(s => s.Exchange === exchange);
        return {
            x: filtered.map((s, i) => i + 1),
            y: filtered.map(s => s.ReturnPct),
            mode: 'markers',
            type: 'scatter',
            name: exchange,
            marker: {
                size: 12,
                opacity: 0.7
            },
            text: filtered.map(s => `${s.Ticker}<br>${s.Name}<br>${s.ReturnPct.toFixed(2)}%`),
            hovertemplate: '%{text}<extra></extra>',
            customdata: filtered.map(s => s.Ticker)
        };
    });
    
    const layout = {
        title: {
            text: `Stock Returns by Exchange - ${metadata?.metric || ''}`,
            font: { size: 20, color: '#2d3748' }
        },
        xaxis: {
            title: 'Rank',
            gridcolor: '#e2e8f0'
        },
        yaxis: {
            title: 'Return (%)',
            gridcolor: '#e2e8f0'
        },
        height: 600,
        paper_bgcolor: 'white',
        plot_bgcolor: '#f7fafc',
        hovermode: 'closest'
    };
    
    Plotly.newPlot(container, traces, layout, {responsive: true});
    
    container.on('plotly_click', function(data) {
        const ticker = data.points[0].customdata;
        viewStockDetail(ticker);
    });
}

function createTreemap(data, metadata, container) {
    const trace = {
        type: 'treemap',
        labels: data.map(s => s.Ticker),
        parents: data.map(s => s.Exchange),
        values: data.map(s => Math.abs(s.ReturnPct)),
        text: data.map(s => `${s.Ticker}<br>${s.ReturnPct.toFixed(2)}%`),
        marker: {
            colors: data.map(s => s.ReturnPct),
            colorscale: [
                [0, '#48bb78'],
                [0.5, '#4299e1'],
                [1, '#9f7aea']
            ],
            line: {
                color: 'white',
                width: 2
            }
        },
        hovertemplate: '<b>%{label}</b><br>Return: %{color:.2f}%<extra></extra>',
        customdata: data.map(s => s.Ticker)
    };
    
    const exchanges = [...new Set(data.map(s => s.Exchange))];
    trace.labels = [...exchanges, ...trace.labels];
    trace.parents = ['', ...trace.parents.map(p => p || '')];
    trace.values = [null, ...trace.values];
    trace.text = ['', ...trace.text];
    trace.customdata = ['', ...trace.customdata];
    
    const layout = {
        title: {
            text: `Stock Returns Treemap - ${metadata?.metric || ''}`,
            font: { size: 20, color: '#2d3748' }
        },
        height: 700,
        paper_bgcolor: 'white'
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true});
    
    container.on('plotly_click', function(data) {
        const ticker = data.points[0].customdata;
        if (ticker) viewStockDetail(ticker);
    });
}

function createSunburst(data, metadata, container) {
    const trace = {
        type: 'sunburst',
        labels: data.map(s => s.Ticker),
        parents: data.map(s => s.Exchange),
        values: data.map(s => Math.abs(s.ReturnPct)),
        text: data.map(s => `${s.ReturnPct.toFixed(2)}%`),
        marker: {
            colors: data.map(s => s.ReturnPct),
            colorscale: [
                [0, '#48bb78'],
                [0.5, '#4299e1'],
                [1, '#9f7aea']
            ],
            line: {
                color: 'white',
                width: 2
            }
        },
        hovertemplate: '<b>%{label}</b><br>Return: %{color:.2f}%<extra></extra>',
        customdata: data.map(s => s.Ticker)
    };
    
    const exchanges = [...new Set(data.map(s => s.Exchange))];
    trace.labels = [...exchanges, ...trace.labels];
    trace.parents = ['', ...trace.parents];
    trace.values = [null, ...trace.values];
    trace.text = ['', ...trace.text];
    trace.customdata = ['', ...trace.customdata];
    
    const layout = {
        title: {
            text: `Stock Returns Sunburst - ${metadata?.metric || ''}`,
            font: { size: 20, color: '#2d3748' }
        },
        height: 700,
        paper_bgcolor: 'white'
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true});
    
    container.on('plotly_click', function(data) {
        const ticker = data.points[0].customdata;
        if (ticker) viewStockDetail(ticker);
    });
}
