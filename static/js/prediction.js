// Prediction page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const urlParams = new URLSearchParams(window.location.search);
    const symbol = urlParams.get('symbol') || localStorage.getItem('lastAnalyzedSymbol');
    
    if (symbol) {
        loadPredictions(symbol);
    } else {
        window.location.href = '/';
    }
});

function loadPredictions(symbol) {
    showLoading('Generating AI predictions...');
    
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `symbol=${symbol}&period=1y`
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showError(data.error);
        } else {
            displayPredictionResults(data);
        }
    })
    .catch(error => {
        hideLoading();
        showError('An error occurred while generating predictions.');
        console.error('Error:', error);
    });
}

function displayPredictionResults(data) {
    document.getElementById('predictionSymbol').textContent = `${data.symbol} - AI Price Predictions`;
    
    displayCurrentStockInfo(data.statistics, data.symbol);
    displayPredictionsTimeline(data.prediction);
    displayPredictionChart(data.prediction, data.statistics);
    displayInsights(data.prediction, data.statistics);
    displayTechnicalSummary(data.statistics);
}

function displayCurrentStockInfo(stats, symbol) {
    const container = document.getElementById('currentStockInfo');
    
    container.innerHTML = `
        <div class="col-md-3 col-6">
            <div class="stock-stat">
                <div class="stat-value">${symbol}</div>
                <div class="stat-label">Symbol</div>
            </div>
        </div>
        <div class="col-md-3 col-6">
            <div class="stock-stat">
                <div class="stat-value ${stats.price_change >= 0 ? 'positive' : 'negative'}">
                    $${stats.current_price}
                </div>
                <div class="stat-label">Current Price</div>
            </div>
        </div>
        <div class="col-md-3 col-6">
            <div class="stock-stat">
                <div class="stat-value ${stats.price_change_percent >= 0 ? 'positive' : 'negative'}">
                    ${stats.price_change_percent >= 0 ? '+' : ''}${stats.price_change_percent}%
                </div>
                <div class="stat-label">Today's Change</div>
            </div>
        </div>
        <div class="col-md-3 col-6">
            <div class="stock-stat">
                <div class="stat-value ${getRSIColor(stats.rsi)}">
                    ${stats.rsi}
                </div>
                <div class="stat-label">RSI</div>
            </div>
        </div>
    `;
}

function displayPredictionsTimeline(prediction) {
    const container = document.getElementById('predictionsTimeline');
    
    if (prediction.error) {
        container.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle"></i>
                ${prediction.error}
            </div>
        `;
        return;
    }
    
    let timelineHtml = '';
    
    prediction.predictions.forEach((pred, index) => {
        const confidence = prediction.confidence[index];
        const confidencePercent = Math.round(confidence * 100);
        const date = prediction.prediction_dates ? prediction.prediction_dates[index] : `Day ${index + 1}`;
        
        timelineHtml += `
            <div class="timeline-item">
                <div class="card mb-3">
                    <div class="card-body">
                        <div class="row align-items-center">
                            <div class="col-md-2">
                                <h5 class="mb-0 ${prediction.trend === 'bullish' ? 'text-success' : 'text-danger'}">
                                    $${pred}
                                </h5>
                                <small class="text-muted">${date}</small>
                            </div>
                            <div class="col-md-6">
                                <div class="confidence-info">
                                    <small class="text-muted">Model Confidence: ${confidencePercent}%</small>
                                    <div class="confidence-bar mt-1">
                                        <div class="confidence-fill ${prediction.trend === 'bullish' ? 'bg-success' : 'bg-danger'}" 
                                             style="width: ${confidencePercent}%"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <span class="badge ${getPredictionTrendClass(prediction.trend)}">
                                    ${getPredictionTrendText(prediction.trend, index)}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = timelineHtml;
}

function displayPredictionChart(prediction, stats) {
    if (prediction.error) return;
    
    const dates = prediction.prediction_dates || 
                 Array.from({length: prediction.prediction_days}, (_, i) => `Day ${i + 1}`);
    
    const trace = {
        x: dates,
        y: prediction.predictions,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Predicted Price',
        line: {
            color: prediction.trend === 'bullish' ? '#28a745' : '#dc3545',
            width: 3
        },
        marker: {
            size: 8,
            color: prediction.trend === 'bullish' ? '#28a745' : '#dc3545'
        }
    };
    
    const currentPriceTrace = {
        x: [dates[0]],
        y: [stats.current_price],
        type: 'scatter',
        mode: 'markers',
        name: 'Current Price',
        marker: {
            size: 12,
            color: '#007bff',
            symbol: 'star'
        }
    };
    
    const layout = {
        title: '7-Day Price Prediction',
        xaxis: {
            title: 'Date'
        },
        yaxis: {
            title: 'Price ($)'
        },
        showlegend: true,
        template: 'plotly_white'
    };
    
    Plotly.newPlot('predictionChart', [trace, currentPriceTrace], layout, {responsive: true});
}

function displayInsights(prediction, stats) {
    const insightContainer = document.getElementById('investmentInsight');
    const riskContainer = document.getElementById('riskAssessment');
    
    if (prediction.error) {
        insightContainer.innerHTML = '<p class="text-muted">Insufficient data for insights.</p>';
        riskContainer.innerHTML = '<p class="text-muted">Risk assessment unavailable.</p>';
        return;
    }
    
    // Investment Insight
    const avgConfidence = prediction.confidence.reduce((a, b) => a + b, 0) / prediction.confidence.length;
    const priceChange = ((prediction.predictions[prediction.predictions.length - 1] - stats.current_price) / stats.current_price) * 100;
    
    let insight = '';
    if (prediction.trend === 'bullish' && avgConfidence > 0.7) {
        insight = 'Strong bullish signals detected. Consider holding or accumulating positions.';
    } else if (prediction.trend === 'bullish' && avgConfidence > 0.5) {
        insight = 'Moderate bullish outlook. Monitor key support levels.';
    } else if (prediction.trend === 'bearish' && avgConfidence > 0.7) {
        insight = 'Strong bearish pressure. Consider risk management strategies.';
    } else {
        insight = 'Mixed signals. Wait for clearer market direction.';
    }
    
    insightContainer.innerHTML = `
        <p class="mb-2">${insight}</p>
        <ul class="small">
            <li>Predicted ${prediction.trend} trend</li>
            <li>Average confidence: ${Math.round(avgConfidence * 100)}%</li>
            <li>Expected change: ${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}%</li>
        </ul>
    `;
    
    // Risk Assessment
    const rsi = stats.rsi;
    let riskLevel = 'Medium';
    let riskColor = 'warning';
    
    if (rsi > 70 || rsi < 30) {
        riskLevel = 'High';
        riskColor = 'danger';
    } else if (avgConfidence < 0.5) {
        riskLevel = 'Medium-High';
        riskColor = 'warning';
    } else if (avgConfidence > 0.7) {
        riskLevel = 'Low-Medium';
        riskColor = 'success';
    }
    
    riskContainer.innerHTML = `
        <div class="mb-3">
            <span class="badge bg-${riskColor}">${riskLevel} Risk</span>
        </div>
        <ul class="small mb-0">
            <li>RSI: ${rsi} ${rsi > 70 ? '(Overbought)' : rsi < 30 ? '(Oversold)' : '(Neutral)'}</li>
            <li>Model confidence: ${Math.round(avgConfidence * 100)}%</li>
            <li>Price volatility: ${stats.price_change_percent >= 0 ? '+' : ''}${Math.abs(stats.price_change_percent)}% today</li>
        </ul>
    `;
}

function displayTechnicalSummary(stats) {
    const container = document.getElementById('technicalSummary');
    
    container.innerHTML = `
        <div class="col-md-3 col-6">
            <div class="stock-stat">
                <div class="stat-value ${getRSIColor(stats.rsi)}">${stats.rsi}</div>
                <div class="stat-label">RSI</div>
                <small class="text-muted">${getRSIInterpretation(stats.rsi)}</small>
            </div>
        </div>
        <div class="col-md-3 col-6">
            <div class="stock-stat">
                <div class="stat-value">$${stats.sma_20}</div>
                <div class="stat-label">SMA 20</div>
                <small class="text-muted">Short-term trend</small>
            </div>
        </div>
        <div class="col-md-3 col-6">
            <div class="stock-stat">
                <div class="stat-value">$${stats.sma_50}</div>
                <div class="stat-label">SMA 50</div>
                <small class="text-muted">Medium-term trend</small>
            </div>
        </div>
        <div class="col-md-3 col-6">
            <div class="stock-stat">
                <div class="stat-value ${stats.price_change_percent >= 0 ? 'positive' : 'negative'}">
                    ${stats.price_change_percent >= 0 ? '+' : ''}${stats.price_change_percent}%
                </div>
                <div class="stat-label">Daily Change</div>
                <small class="text-muted">Market sentiment</small>
            </div>
        </div>
    `;
}

function getRSIColor(rsi) {
    if (rsi > 70) return 'negative';
    if (rsi < 30) return 'positive';
    return 'neutral';
}

function getRSIInterpretation(rsi) {
    if (rsi > 70) return 'Overbought';
    if (rsi < 30) return 'Oversold';
    return 'Neutral';
}

function getPredictionTrendClass(trend) {
    return trend === 'bullish' ? 'bg-success' : 'bg-danger';
}

function getPredictionTrendText(trend, dayIndex) {
    const actions = trend === 'bullish' 
        ? ['Strong Start', 'Continuing Rise', 'Momentum Builds', 'Peak Expected', 'Consolidation', 'New Highs', 'Bullish Close']
        : ['Initial Drop', 'Downward Pressure', 'Bearish Momentum', 'Support Test', 'Recovery Attempt', 'Lower Lows', 'Bearish Close'];
    
    return actions[dayIndex] || (trend === 'bullish' ? 'Bullish' : 'Bearish');
}

function goToAnalysis() {
    const symbol = localStorage.getItem('lastAnalyzedSymbol');
    window.location.href = `/analysis?symbol=${symbol}`;
}

function analyzeNewStock() {
    window.location.href = '/';
}

function showLoading(message) {
    // Create a simple loading indicator for prediction page
    const container = document.getElementById('predictionsTimeline');
    container.innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>${message}</p>
        </div>
    `;
}

function hideLoading() {
    // Loading is handled by replacing content
}

function showError(message) {
    const container = document.getElementById('predictionsTimeline');
    container.innerHTML = `
        <div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle"></i> ${message}
        </div>
        <div class="text-center mt-3">
            <button class="btn btn-primary" onclick="analyzeNewStock()">
                Try Another Stock
            </button>
        </div>
    `;
}