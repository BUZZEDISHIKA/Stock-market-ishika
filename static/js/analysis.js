// Analysis page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('Analysis page loaded');
    
    // Get stock data from URL parameters or localStorage
    const urlParams = new URLSearchParams(window.location.search);
    const symbol = urlParams.get('symbol') || localStorage.getItem('lastAnalyzedSymbol') || 'AAPL';
    const period = urlParams.get('period') || '1y';
    
    console.log('Symbol:', symbol, 'Period:', period);
    
    if (symbol) {
        loadStockAnalysis(symbol, period);
    } else {
        // Redirect to home if no symbol
        window.location.href = '/';
    }
});

function loadStockAnalysis(symbol, period) {
    console.log('Loading analysis for:', symbol);
    showLoading('Analyzing stock data...');
    
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `symbol=${symbol}&period=${period}`
    })
    .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.status);
        }
        return response.json();
    })
    .then(data => {
        console.log('Data received:', data);
        hideLoading();
        
        if (data.error) {
            showError(data.error);
        } else {
            displayAnalysisResults(data);
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        hideLoading();
        showError('An error occurred while analyzing the stock: ' + error.message);
    });
}

function displayAnalysisResults(data) {
    console.log('Displaying analysis results:', data);
    
    // Update page title and header
    document.getElementById('stockSymbol').textContent = `${data.symbol} - Comprehensive Analysis`;
    document.getElementById('analysisPeriod').textContent = `Period: ${data.period || '1y'}`;
    
    // Display statistics
    displayStockStatistics(data.statistics, data.symbol);
    
    // Display charts
    displayCharts(data.plots);
    
    // Display predictions
    displayPredictions(data.prediction, data.statistics);
    
    // Display sentiment analysis
    displaySentimentAnalysis(data.sentiment);
    
    // Store last analyzed symbol
    localStorage.setItem('lastAnalyzedSymbol', data.symbol);
}

function displayStockStatistics(stats, symbol) {
    const statsContainer = document.getElementById('stockStats');
    console.log('Displaying stats:', stats);
    
    if (!stats || Object.keys(stats).length === 0) {
        statsContainer.innerHTML = '<div class="col-12 text-center"><p class="text-warning">No statistics available</p></div>';
        return;
    }
    
    const statsHtml = `
        <div class="col-md-2 col-6 mb-3">
            <div class="stock-stat">
                <div class="stat-value ${stats.price_change >= 0 ? 'positive' : 'negative'}">
                    $${stats.current_price || 'N/A'}
                </div>
                <div class="stat-label">Current Price</div>
            </div>
        </div>
        <div class="col-md-2 col-6 mb-3">
            <div class="stock-stat">
                <div class="stat-value ${stats.price_change >= 0 ? 'positive' : 'negative'}">
                    ${stats.price_change >= 0 ? '+' : ''}$${stats.price_change || 0}
                </div>
                <div class="stat-label">Price Change</div>
            </div>
        </div>
        <div class="col-md-2 col-6 mb-3">
            <div class="stock-stat">
                <div class="stat-value ${stats.price_change_percent >= 0 ? 'positive' : 'negative'}">
                    ${stats.price_change_percent >= 0 ? '+' : ''}${stats.price_change_percent || 0}%
                </div>
                <div class="stat-label">Change %</div>
            </div>
        </div>
        <div class="col-md-2 col-6 mb-3">
            <div class="stock-stat">
                <div class="stat-value ${getRSIColor(stats.rsi)}">
                    ${stats.rsi || 50}
                </div>
                <div class="stat-label">RSI</div>
            </div>
        </div>
        <div class="col-md-2 col-6 mb-3">
            <div class="stock-stat">
                <div class="stat-value">$${stats.sma_20 || stats.current_price || 'N/A'}</div>
                <div class="stat-label">SMA 20</div>
            </div>
        </div>
        <div class="col-md-2 col-6 mb-3">
            <div class="stock-stat">
                <div class="stat-value">$${stats.sma_50 || stats.current_price || 'N/A'}</div>
                <div class="stat-label">SMA 50</div>
            </div>
        </div>
    `;
    
    statsContainer.innerHTML = statsHtml;
}

function getRSIColor(rsi) {
    if (rsi > 70) return 'negative';
    if (rsi < 30) return 'positive';
    return 'neutral';
}

function displayCharts(plots) {
    console.log('Displaying charts:', plots);
    
    // Render Price Chart
    if (plots && plots.price_chart) {
        try {
            const priceChartData = JSON.parse(plots.price_chart);
            Plotly.newPlot('priceChart', priceChartData.data, priceChartData.layout, {
                responsive: true,
                displayModeBar: true
            });
        } catch (e) {
            console.error('Error rendering price chart:', e);
            document.getElementById('priceChart').innerHTML = '<p class="text-danger">Error loading price chart</p>';
        }
    } else {
        document.getElementById('priceChart').innerHTML = '<p class="text-warning">No price chart data available</p>';
    }
    
    // Render RSI Chart
    if (plots && plots.rsi_chart) {
        try {
            const rsiChartData = JSON.parse(plots.rsi_chart);
            Plotly.newPlot('rsiChart', rsiChartData.data, rsiChartData.layout, {
                responsive: true,
                displayModeBar: true
            });
        } catch (e) {
            console.error('Error rendering RSI chart:', e);
            document.getElementById('rsiChart').innerHTML = '<p class="text-danger">Error loading RSI chart</p>';
        }
    } else {
        document.getElementById('rsiChart').innerHTML = '<p class="text-warning">No RSI chart data available</p>';
    }
    
    // Render Volume Chart
    if (plots && plots.volume_chart) {
        try {
            const volumeChartData = JSON.parse(plots.volume_chart);
            Plotly.newPlot('volumeChart', volumeChartData.data, volumeChartData.layout, {
                responsive: true,
                displayModeBar: true
            });
        } catch (e) {
            console.error('Error rendering volume chart:', e);
            document.getElementById('volumeChart').innerHTML = '<p class="text-danger">Error loading volume chart</p>';
        }
    } else {
        document.getElementById('volumeChart').innerHTML = '<p class="text-warning">No volume chart data available</p>';
    }
}

function displayPredictions(prediction, stats) {
    const predictionsContainer = document.getElementById('predictionsContent');
    console.log('Displaying predictions:', prediction);
    
    if (!prediction || prediction.error) {
        predictionsContainer.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle"></i>
                ${prediction?.error || 'No prediction data available'}
            </div>
        `;
        return;
    }
    
    const predictionsHtml = `
        <div class="row">
            <div class="col-md-8">
                <h6>Next ${prediction.prediction_days || 7} Days Forecast:</h6>
                <div class="row">
                    ${prediction.predictions.map((pred, index) => `
                        <div class="col-md-3 col-6 mb-3">
                            <div class="card prediction-card ${prediction.trend === 'bullish' ? 'border-success' : 'border-danger'}">
                                <div class="card-body text-center p-2">
                                    <small class="text-muted">Day ${index + 1}</small>
                                    <h6 class="mb-1 ${prediction.trend === 'bullish' ? 'text-success' : 'text-danger'}">
                                        $${pred}
                                    </h6>
                                    <div class="confidence-bar mt-1">
                                        <div class="confidence-fill bg-${prediction.trend === 'bullish' ? 'success' : 'danger'}" 
                                             style="width: ${(prediction.confidence?.[index] || 0.5) * 100}%"></div>
                                    </div>
                                    <small class="text-muted">${Math.round((prediction.confidence?.[index] || 0.5) * 100)}% conf</small>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-light">
                    <div class="card-body">
                        <h6>Prediction Summary</h6>
                        <p class="mb-2">
                            <strong>Trend:</strong> 
                            <span class="badge ${prediction.trend === 'bullish' ? 'bg-success' : 'bg-danger'}">
                                ${(prediction.trend || 'neutral').toUpperCase()}
                            </span>
                        </p>
                        <p class="mb-2">
                            <strong>Current Price:</strong> $${stats?.current_price || 'N/A'}
                        </p>
                        <p class="mb-0">
                            <strong>Predicted Change:</strong> 
                            <span class="${prediction.predicted_change_percent >= 0 ? 'positive' : 'negative'}">
                                ${prediction.predicted_change_percent >= 0 ? '+' : ''}${prediction.predicted_change_percent || 0}%
                            </span>
                        </p>
                    </div>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <a href="/prediction?symbol=${localStorage.getItem('lastAnalyzedSymbol')}" class="btn btn-outline-primary btn-sm">
                <i class="fas fa-chart-line"></i> View Detailed Predictions
            </a>
        </div>
    `;
    
    predictionsContainer.innerHTML = predictionsHtml;
}

function displaySentimentAnalysis(sentiment) {
    const sentimentContainer = document.getElementById('sentimentContent');
    const sentimentSource = document.getElementById('sentimentSource');
    
    console.log('Displaying sentiment:', sentiment);
    
    sentimentSource.textContent = sentiment?.source === 'real_news' ? 'Live News Data' : 'Market Analysis';
    
    if (!sentiment) {
        sentimentContainer.innerHTML = '<p class="text-warning">No sentiment data available</p>';
        return;
    }
    
    const sentimentHtml = `
        <div class="row align-items-center">
            <div class="col-md-3 text-center">
                <div class="mb-3">
                    <span class="badge ${getSentimentClass(sentiment.overall_sentiment)} sentiment-badge fs-6">
                        ${(sentiment.overall_sentiment || 'neutral').toUpperCase()}
                    </span>
                </div>
                <div class="sentiment-score">
                    <h4 class="${getSentimentColor(sentiment.sentiment_score)}">
                        ${sentiment.sentiment_score > 0 ? '+' : ''}${sentiment.sentiment_score || 0}
                    </h4>
                    <small class="text-muted">Sentiment Score</small>
                </div>
                <div class="mt-2">
                    <small class="text-muted">
                        Confidence: ${Math.round((sentiment.confidence || 0.7) * 100)}%
                    </small>
                </div>
            </div>
            <div class="col-md-9">
                <h6>Recent Market News:</h6>
                <div class="news-headlines">
                    ${(sentiment.sample_headlines || ['No news available']).map(headline => `
                        <div class="news-item mb-2 p-2 border rounded">
                            <i class="fas fa-newspaper text-primary me-2"></i>
                            ${headline}
                        </div>
                    `).join('')}
                </div>
                ${sentiment.articles_count ? `
                    <small class="text-muted">
                        Analyzed ${sentiment.articles_count} news articles
                    </small>
                ` : ''}
            </div>
        </div>
    `;
    
    sentimentContainer.innerHTML = sentimentHtml;
}

function getSentimentClass(sentiment) {
    switch(sentiment) {
        case 'positive': return 'bg-success';
        case 'negative': return 'bg-danger';
        default: return 'bg-secondary';
    }
}

function getSentimentColor(score) {
    if (score > 0.1) return 'positive';
    if (score < -0.1) return 'negative';
    return 'neutral';
}

function analyzeNewStock() {
    window.location.href = '/';
}

function showLoading(message) {
    document.getElementById('loadingMessage').textContent = message;
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    loadingModal.show();
}

function hideLoading() {
    const loadingModal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
    if (loadingModal) {
        loadingModal.hide();
    }
}

function showError(message) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            <strong>Error:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
        <div class="text-center mt-4">
            <button class="btn btn-primary" onclick="analyzeNewStock()">
                Try Another Stock
            </button>
        </div>
    `;
}