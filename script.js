// Global variables
let currentStartupData = null;
let agentAnalyses = {
    investor: null,
    researcher: null,
    user: null
};

// API configuration
const API_BASE_URL = 'http://localhost:8002';
const API_ENDPOINTS = {
    analyze: '/analyze',
    agents: '/agents',
    health: '/health',
    status: '/status'
};

// DOM elements
const startupForm = document.getElementById('startupForm');
const filesInput = document.getElementById('files');
const fileList = document.getElementById('fileList');
const agentsSection = document.getElementById('agentsSection');
const collaborationSection = document.getElementById('collaborationSection');
const recommendationSection = document.getElementById('recommendationSection');
const resultsSection = document.getElementById('resultsSection');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('StartupAI Multi-Agent Application Initialized');
    
    // Add form submission handler
    startupForm.addEventListener('submit', handleFormSubmission);
    // Upload preview
    if (filesInput) {
        filesInput.addEventListener('change', () => {
            if (!filesInput.files) return;
            const items = Array.from(filesInput.files).map(f => `• ${f.name} (${Math.round(f.size/1024)} KB)`).join('<br/>');
            if (fileList) fileList.innerHTML = items;
        });
    }
    
    // Add smooth scrolling for better UX
    addSmoothScrolling();
    
    // Check API health
    checkAPIHealth();
});

// Check API health status
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.health}`);
        if (response.ok) {
            console.log('Multi-Agent API is healthy');
            // Show API status indicator
            showAPIStatus('connected');
        } else {
            throw new Error('API health check failed');
        }
    } catch (error) {
        console.warn('Multi-Agent API not available, falling back to simulation mode');
        showAPIStatus('disconnected');
    }
}

// Show API connection status
function showAPIStatus(status) {
    const statusIndicator = document.createElement('div');
    statusIndicator.id = 'apiStatus';
    statusIndicator.className = `api-status ${status}`;
    statusIndicator.innerHTML = `
        <i class="fas fa-${status === 'connected' ? 'check-circle' : 'exclamation-triangle'}"></i>
        <span>API ${status === 'connected' ? 'Connected' : 'Disconnected'}</span>
    `;
    
    // Add to header
    const header = document.querySelector('.header-content');
    if (!document.getElementById('apiStatus')) {
        header.appendChild(statusIndicator);
    }
}

// Handle form submission
async function handleFormSubmission(event) {
    event.preventDefault();
    
    // Get form data
    const formData = new FormData(startupForm);
    currentStartupData = {
        startupName: formData.get('startupName'),
        briefDescription: formData.get('briefDescription') || '',
        problemStatement: formData.get('problemStatement'),
        solution: formData.get('solution'),
        targetMarket: formData.get('targetMarket'),
        businessModel: formData.get('businessModel'),
        fundingStage: formData.get('fundingStage') || 'Pre-Seed',
        teamSize: formData.get('teamSize') || '1-2 People',
        competitiveAdvantage: formData.get('competitiveAdvantage')
    };
    
    // Show agents section and scroll to it
    showAgentsSection();
    scrollToSection(agentsSection);
    
    // Start the AI evaluation process
    await startAIEvaluation();
}

// Show agents section
function showAgentsSection() {
    agentsSection.style.display = 'block';
    agentsSection.classList.add('fade-in');
}

// Start AI evaluation process
async function startAIEvaluation() {
    try {
        // Check if API is available
        const apiAvailable = await checkAPIAvailability();
        
        if (apiAvailable) {
            // Use real multi-agent API
            await runRealAIEvaluation();
        } else {
            // Fall back to simulation mode
            await runSimulatedEvaluation();
        }
        
    } catch (error) {
        console.error('Error during AI evaluation:', error);
        showError('An error occurred during the AI evaluation process.');
        
        // Fall back to simulation
        await runSimulatedEvaluation();
    }
}

// Check API availability
async function checkAPIAvailability() {
    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.health}`);
        return response.ok;
    } catch (error) {
        return false;
    }
}

// Run real AI evaluation using the multi-agent API
async function runRealAIEvaluation() {
    try {
        console.log('Starting real multi-agent evaluation...');
        
        // Prepare startup data for API
        const startupRequest = {
            name: currentStartupData.startupName,
            problem_statement: currentStartupData.problemStatement,
            solution: currentStartupData.solution,
            target_market: currentStartupData.targetMarket,
            business_model: currentStartupData.businessModel,
            competitive_advantage: currentStartupData.competitiveAdvantage
        };
        
        // Submit analysis request
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.analyze}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ topic: currentStartupData.startupName })
        });
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.status === 'success' && result.data) {
            // Process real agent analyses
            await processRealAgentAnalyses(result.data);
        } else {
            throw new Error(result.error || 'Analysis failed');
        }
        
    } catch (error) {
        console.error('Real AI evaluation failed:', error);
        throw error;
    }
}

// Process real agent analyses from API
async function processRealAgentAnalyses(analysisData) {
    try {
        const { agent_analyses } = analysisData;
        let allHaveLLM = true;
        
        // Process each agent analysis
        for (const [agentId, analysis] of Object.entries(agent_analyses)) {
            const agentName = agentId; // investor, researcher, user
            const loadingElement = document.getElementById(`${agentName}Loading`);
            const analysisElement = document.getElementById(`${agentName}Analysis`);
            
            // Hide loading, show analysis
            loadingElement.style.display = 'none';
            analysisElement.style.display = 'block';
            analysisElement.classList.add('slide-up');
            
            // Display real analysis
            const formattedAnalysis = formatRealAnalysis(analysis);
            analysisElement.innerHTML = formattedAnalysis;
            
            // Store analysis
            agentAnalyses[agentName] = analysis;
            if (!analysis.llm_analysis) {
                allHaveLLM = false;
            }
            
            // Add delay for better UX
            await delay(1000);
        }
        
        // Only proceed to discussion/recommendation if all agents have real LLM output
        if (allHaveLLM) {
            await delay(2000);
            await showRealCollaborativeDiscussion(agent_analyses);
            await delay(2000);
            await showRealFinalRecommendation(analysisData);
            await delay(1000);
            showRealResultsSummary(analysisData);
        } else {
            // Hide downstream sections when LLM output is missing
            collaborationSection.style.display = 'none';
            recommendationSection.style.display = 'none';
            resultsSection.style.display = 'none';
        }
        
    } catch (error) {
        console.error('Error processing real agent analyses:', error);
        throw error;
    }
}

// Format real analysis for display
function formatRealAnalysis(analysis) {
    const { agent, role, llm_analysis, relevant_documents, document_previews } = analysis;
    
    let documentsHtml = '';
    if (document_previews && document_previews.length > 0) {
        documentsHtml = '<h4>📚 Relevant Documents Found</h4><ul>';
        for (const doc of document_previews.slice(0, 3)) {
            documentsHtml += `<li><strong>${doc.source.split('\\').pop()}</strong> (relevance: ${(doc.relevance * 100).toFixed(1)}%)</li>`;
        }
        documentsHtml += '</ul>';
    }
    
    return `
        <h4>${role} Analysis</h4>
        <p><strong>Documents Analyzed:</strong> ${relevant_documents || 0}</p>
        
        ${documentsHtml}
        
        <h4>🧠 AI Analysis</h4>
        <div class="analysis-text">
            <p>${llm_analysis ? llm_analysis : '<em>Awaiting LLM analysis from server...</em>'}</p>
        </div>
    `;
}

// Show real collaborative discussion
async function showRealCollaborativeDiscussion(agentAnalyses) {
    collaborationSection.style.display = 'block';
    collaborationSection.classList.add('fade-in');
    
    const loadingElement = document.getElementById('collaborationLoading');
    const discussionElement = document.getElementById('discussionThread');
    
    // Simulate discussion time
    await delay(2000);
    
    // Hide loading, show discussion
    loadingElement.style.display = 'none';
    discussionElement.style.display = 'block';
    discussionElement.classList.add('slide-up');
    
    // Generate discussion based on real analyses
    const discussion = generateRealCollaborativeDiscussion(agentAnalyses);
    discussionElement.innerHTML = discussion;
    
    scrollToSection(collaborationSection);
}

// Generate real collaborative discussion
function generateRealCollaborativeDiscussion(agentAnalyses) {
    let discussionHtml = '';
    
    for (const [agentId, analysis] of Object.entries(agentAnalyses)) {
        const { agent, role, llm_analysis } = analysis;
        
        // Extract key points from the LLM analysis
        const analysisPreview = llm_analysis ? llm_analysis.substring(0, 200) + '...' : 'Analysis completed';
        
        discussionHtml += `
            <div class="discussion-item">
                <div class="discussion-header">
                    <div class="discussion-avatar ${agentId}">
                        <i class="fas fa-${getRoleIcon(agentId)}"></i>
                    </div>
                    <div class="discussion-author">${agent}</div>
                </div>
                <div class="discussion-content">
                    "Based on my ${role} analysis: <strong>${analysisPreview}</strong>"
                </div>
            </div>
        `;
    }
    
    // Add consensus building discussion
    discussionHtml += `
        <div class="discussion-item">
            <div class="discussion-header">
                <div class="discussion-avatar investor">
                    <i class="fas fa-chart-line"></i>
                </div>
                <div class="discussion-author">System Consensus</div>
            </div>
            <div class="discussion-content">
                "After reviewing all perspectives, we've identified the key success factors and areas requiring attention. 
                The consensus approach will balance investment potential, technical feasibility, and user experience considerations."
            </div>
        </div>
    `;
    
    return discussionHtml;
}

// Get role icon
function getRoleIcon(role) {
    const icons = {
        'investor': 'chart-line',
        'researcher': 'microscope',
        'user': 'user'
    };
    return icons[role] || 'robot';
}

// Show real final recommendation
async function showRealFinalRecommendation(analysisData) {
    recommendationSection.style.display = 'block';
    recommendationSection.classList.add('fade-in');
    
    const loadingElement = document.getElementById('recommendationLoading');
    const summaryElement = document.getElementById('recommendationSummary');
    
    // Simulate recommendation generation time
    await delay(2000);
    
    // Hide loading, show recommendation
    loadingElement.style.display = 'none';
    summaryElement.style.display = 'block';
    summaryElement.classList.add('slide-up');
    
    // Display real analysis summary
    const recommendation = formatRealRecommendation(analysisData);
    summaryElement.innerHTML = recommendation;
    
    scrollToSection(recommendationSection);
}

// Format real recommendation for display
function formatRealRecommendation(analysisData) {
    const { topic, agent_analyses, summary } = analysisData;
    
    return `
        <h4>Final Assessment</h4>
        <p><strong>Topic Analyzed:</strong> ${topic}</p>
        
        <h4>Analysis Summary</h4>
        <ul>
            <li><strong>Total Agents:</strong> ${summary.total_agents}</li>
            <li><strong>Documents Analyzed:</strong> ${summary.total_relevant_documents}</li>
            <li><strong>Analysis Complete:</strong> ${summary.analysis_complete ? 'Yes' : 'No'}</li>
        </ul>
        
        <h4>Agent Insights</h4>
        <ul>
            <li><strong>Investor:</strong> ${agent_analyses.investor?.llm_analysis?.substring(0, 100) || 'Analysis pending'}...</li>
            <li><strong>Researcher:</strong> ${agent_analyses.researcher?.llm_analysis?.substring(0, 100) || 'Analysis pending'}...</li>
            <li><strong>User:</strong> ${agent_analyses.user?.llm_analysis?.substring(0, 100) || 'Analysis pending'}...</li>
        </ul>
        
        <h4>Recommendation</h4>
        <p>Based on the multi-agent analysis, this startup idea shows potential across different perspectives. Review the detailed analyses above for specific insights and recommendations.</p>
    `;
}

// Show real results summary
function showRealResultsSummary(analysisData) {
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');
    
    // Calculate and display summary metrics
    displayRealSummaryMetrics(analysisData);
    
    scrollToSection(resultsSection);
}

// Display real summary metrics
function displayRealSummaryMetrics(analysisData) {
    const { summary, agent_analyses } = analysisData;
    
    // Calculate a simple score based on document relevance
    const avgRelevance = Object.values(agent_analyses).reduce((sum, analysis) => {
        if (analysis.document_previews && analysis.document_previews.length > 0) {
            const avg = analysis.document_previews.reduce((s, doc) => s + doc.relevance, 0) / analysis.document_previews.length;
            return sum + avg;
        }
        return sum;
    }, 0) / Object.keys(agent_analyses).length;
    
    const overallScore = Math.round(avgRelevance * 10);
    
    // Update score elements
    document.getElementById('overallScore').textContent = `${overallScore}/10`;
    document.getElementById('overallScore').className = `score ${getScoreClass(overallScore)}`;
    
    // Set other metrics based on analysis quality
    document.getElementById('riskLevel').textContent = overallScore >= 7 ? 'Low' : overallScore >= 5 ? 'Medium' : 'High';
    document.getElementById('riskLevel').className = `risk-level ${getScoreClass(overallScore >= 7 ? 9 : overallScore >= 5 ? 6 : 3)}`;
    
    document.getElementById('marketPotential').textContent = overallScore >= 7 ? 'High' : overallScore >= 5 ? 'Medium' : 'Low';
    document.getElementById('marketPotential').className = `market-potential ${getScoreClass(overallScore >= 7 ? 9 : overallScore >= 5 ? 6 : 3)}`;
    
    document.getElementById('feasibility').textContent = overallScore >= 7 ? 'High' : overallScore >= 5 ? 'Medium' : 'Low';
    document.getElementById('feasibility').className = `feasibility ${getScoreClass(overallScore >= 7 ? 9 : overallScore >= 5 ? 6 : 3)}`;
}

// Fallback to simulated evaluation
async function runSimulatedEvaluation() {
    console.log('Running simulated evaluation...');
    
    try {
        // Start individual agent analyses
        await Promise.all([
            runInvestorAnalysis(),
            runResearcherAnalysis(),
            runUserAnalysis()
        ]);
        
        // Wait a bit for user to read individual analyses
        await delay(2000);
        
        // Start collaborative discussion
        await startCollaborativeDiscussion();
        
        // Wait a bit more
        await delay(2000);
        
        // Generate final recommendation
        await generateFinalRecommendation();
        
        // Show results summary
        await delay(1000);
        showResultsSummary();
        
    } catch (error) {
        console.error('Error during simulated evaluation:', error);
        showError('An error occurred during the evaluation process.');
    }
}

// Simulated Investor Agent Analysis (fallback)
async function runInvestorAnalysis() {
    const loadingElement = document.getElementById('investorLoading');
    const analysisElement = document.getElementById('investorAnalysis');
    
    // Simulate analysis time
    await delay(3000 + Math.random() * 2000);
    
    // Hide loading, show analysis
    loadingElement.style.display = 'none';
    analysisElement.style.display = 'block';
    analysisElement.classList.add('slide-up');
    
    // Generate investor analysis
    const analysis = generateInvestorAnalysis();
    analysisElement.innerHTML = analysis;
    
    agentAnalyses.investor = analysis;
}

// Simulated Researcher Agent Analysis (fallback)
async function runResearcherAnalysis() {
    const loadingElement = document.getElementById('researcherLoading');
    const analysisElement = document.getElementById('researcherAnalysis');
    
    // Simulate analysis time
    await delay(4000 + Math.random() * 2000);
    
    // Hide loading, show analysis
    loadingElement.style.display = 'none';
    analysisElement.style.display = 'block';
    analysisElement.classList.add('slide-up');
    
    // Generate researcher analysis
    const analysis = generateResearcherAnalysis();
    analysisElement.innerHTML = analysis;
    
    agentAnalyses.researcher = analysis;
}

// Simulated User Agent Analysis (fallback)
async function runUserAnalysis() {
    const loadingElement = document.getElementById('userLoading');
    const analysisElement = document.getElementById('userAnalysis');
    
    // Simulate analysis time
    await delay(3500 + Math.random() * 2000);
    
    // Hide loading, show analysis
    loadingElement.style.display = 'none';
    analysisElement.style.display = 'block';
    analysisElement.classList.add('slide-up');
    
    // Generate user analysis
    const analysis = generateUserAnalysis();
    analysisElement.innerHTML = analysis;
    
    agentAnalyses.user = analysis;
}

// Simulated collaborative discussion (fallback)
async function startCollaborativeDiscussion() {
    collaborationSection.style.display = 'block';
    collaborationSection.classList.add('fade-in');
    
    const loadingElement = document.getElementById('collaborationLoading');
    const discussionElement = document.getElementById('discussionThread');
    
    // Simulate discussion time
    await delay(4000 + Math.random() * 2000);
    
    // Hide loading, show discussion
    loadingElement.style.display = 'none';
    discussionElement.style.display = 'block';
    discussionElement.classList.add('slide-up');
    
    // Generate collaborative discussion
    const discussion = generateCollaborativeDiscussion();
    discussionElement.innerHTML = discussion;
    
    scrollToSection(collaborationSection);
}

// Simulated final recommendation (fallback)
async function generateFinalRecommendation() {
    recommendationSection.style.display = 'block';
    recommendationSection.classList.add('fade-in');
    
    const loadingElement = document.getElementById('recommendationLoading');
    const summaryElement = document.getElementById('recommendationSummary');
    
    // Simulate recommendation generation time
    await delay(3000 + Math.random() * 2000);
    
    // Hide loading, show recommendation
    loadingElement.style.display = 'none';
    summaryElement.style.display = 'block';
    summaryElement.classList.add('slide-up');
    
    // Generate final recommendation
    const recommendation = generateFinalRecommendationContent();
    summaryElement.innerHTML = recommendation;
    
    scrollToSection(recommendationSection);
}

// Show results summary (fallback)
function showResultsSummary() {
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');
    
    // Calculate and display summary metrics
    displaySummaryMetrics();
    
    scrollToSection(resultsSection);
}

// Generate Investor Analysis (fallback)
function generateInvestorAnalysis() {
    const startup = currentStartupData;
    
    // Analyze business model strength
    const businessModelScore = analyzeBusinessModel(startup.businessModel);
    const marketSize = estimateMarketSize(startup.targetMarket);
    const competitiveAdvantage = analyzeCompetitiveAdvantage(startup.competitiveAdvantage);
    
    let investmentRecommendation = 'Consider with modifications';
    let riskLevel = 'Medium';
    let expectedROI = '15-25%';
    
    if (businessModelScore > 7 && marketSize > 1000000000) {
        investmentRecommendation = 'Strong investment potential';
        riskLevel = 'Low';
        expectedROI = '25-40%';
    } else if (businessModelScore < 5 || marketSize < 100000000) {
        investmentRecommendation = 'High risk, not recommended';
        riskLevel = 'High';
        expectedROI = '5-15%';
    }
    
    return `
        <h4>Investment Analysis</h4>
        <p><strong>Recommendation:</strong> ${investmentRecommendation}</p>
        <p><strong>Risk Level:</strong> ${riskLevel}</p>
        <p><strong>Expected ROI:</strong> ${expectedROI}</p>
        
        <h4>Key Strengths</h4>
        <ul>
            <li>${startup.solution.length > 100 ? 'Comprehensive solution approach' : 'Clear problem identification'}</li>
            <li>${startup.targetMarket.includes('B2B') ? 'B2B market with recurring revenue potential' : 'Targeted market segment'}</li>
            <li>${startup.competitiveAdvantage.length > 50 ? 'Strong competitive differentiation' : 'Clear value proposition'}</li>
        </ul>
        
        <h4>Areas of Concern</h4>
        <ul>
            <li>${startup.businessModel.length < 50 ? 'Business model needs more detail' : 'Revenue streams could be diversified'}</li>
            <li>${startup.problemStatement.length < 100 ? 'Problem statement could be more compelling' : 'Market validation needed'}</li>
        </ul>
        
        <h4>Market Opportunity</h4>
        <p>Estimated market size: $${(marketSize / 1000000).toFixed(1)}M</p>
        <p>Market growth potential: ${marketSize > 1000000000 ? 'High' : marketSize > 500000000 ? 'Medium' : 'Limited'}</p>
    `;
}

// Generate Researcher Analysis (fallback)
function generateResearcherAnalysis() {
    const startup = currentStartupData;
    
    // Analyze technical feasibility
    const technicalFeasibility = analyzeTechnicalFeasibility(startup.solution);
    const marketResearch = analyzeMarketResearch(startup.targetMarket);
    const competitiveLandscape = analyzeCompetitiveLandscape(startup.competitiveAdvantage);
    
    return `
        <h4>Technical & Market Research</h4>
        <p><strong>Technical Feasibility:</strong> ${technicalFeasibility}</p>
        <p><strong>Market Validation:</strong> ${marketResearch.validation}</p>
        <p><strong>Competitive Landscape:</strong> ${competitiveLandscape.intensity}</p>
        
        <h4>Market Insights</h4>
        <ul>
            <li>${marketResearch.trends}</li>
            <li>${marketResearch.growth}</li>
            <li>${marketResearch.challenges}</li>
        </ul>
        
        <h4>Technical Considerations</h4>
        <ul>
            <li>${technicalFeasibility.includes('High') ? 'Solution leverages proven technologies' : 'May require innovative technical approach'}</li>
            <li>${startup.solution.includes('AI') || startup.solution.includes('machine learning') ? 'AI/ML integration potential' : 'Traditional solution approach'}</li>
            <li>${startup.solution.length > 150 ? 'Comprehensive technical scope' : 'Focused technical implementation'}</li>
        </ul>
        
        <h4>Research Recommendations</h4>
        <ul>
            <li>Conduct user interviews with target market</li>
            <li>Analyze competitor pricing strategies</li>
            <li>Validate technical assumptions with experts</li>
        </ul>
    `;
}

// Generate User Analysis (fallback)
function generateUserAnalysis() {
    const startup = currentStartupData;
    
    // Analyze user experience potential
    const userExperience = analyzeUserExperience(startup.solution);
    const adoptionPotential = analyzeAdoptionPotential(startup.targetMarket);
    const userValue = analyzeUserValue(startup.solution, startup.problemStatement);
    
    return `
        <h4>User Experience & Adoption</h4>
        <p><strong>User Experience Quality:</strong> ${userExperience.quality}</p>
        <p><strong>Adoption Potential:</strong> ${adoptionPotential.level}</p>
        <p><strong>User Value Proposition:</strong> ${userValue.strength}</p>
        
        <h4>User Benefits</h4>
        <ul>
            <li>${userValue.benefits[0]}</li>
            <li>${userValue.benefits[1]}</li>
            <li>${userValue.benefits[2]}</li>
        </ul>
        
        <h4>Adoption Factors</h4>
        <ul>
            <li>${adoptionPotential.factors[0]}</li>
            <li>${adoptionPotential.factors[1]}</li>
            <li>${adoptionPotential.factors[2]}</li>
        </ul>
        
        <h4>User Experience Considerations</h4>
        <ul>
            <li>${userExperience.considerations[0]}</li>
            <li>${userExperience.considerations[1]}</li>
            <li>${userExperience.considerations[2]}</li>
        </ul>
        
        <h4>Recommendations</h4>
        <ul>
            <li>Conduct user testing early in development</li>
            <li>Focus on intuitive user interface design</li>
            <li>Implement feedback collection mechanisms</li>
        </ul>
    `;
}

// Generate Collaborative Discussion (fallback)
function generateCollaborativeDiscussion() {
    return `
        <div class="discussion-item">
            <div class="discussion-header">
                <div class="discussion-avatar investor">
                    <i class="fas fa-chart-line"></i>
                </div>
                <div class="discussion-author">Investor Agent</div>
            </div>
            <div class="discussion-content">
                "Based on my financial analysis, this startup shows promising market potential. However, I'm concerned about the business model clarity. We need to see more detailed revenue projections and customer acquisition costs."
            </div>
        </div>
        
        <div class="discussion-item">
            <div class="discussion-header">
                <div class="discussion-avatar researcher">
                    <i class="fas fa-microscope"></i>
                </div>
                <div class="discussion-author">Researcher Agent</div>
            </div>
            <div class="discussion-content">
                "I agree with the Investor's concerns. My market research shows strong demand, but the competitive landscape is crowded. The technical solution is feasible, but differentiation will be key to success."
            </div>
        </div>
        
        <div class="discussion-item">
            <div class="discussion-header">
                <div class="discussion-avatar user">
                    <i class="fas fa-user"></i>
                </div>
                <div class="discussion-author">User Agent</div>
            </div>
            <div class="discussion-content">
                "From a user perspective, the solution addresses a real pain point. However, the user experience needs to be significantly better than existing alternatives to drive adoption. I recommend focusing on UX design."
            </div>
        </div>
        
        <div class="discussion-item">
            <div class="discussion-header">
                <div class="discussion-avatar investor">
                    <i class="fas fa-chart-line"></i>
                </div>
                <div class="discussion-author">Investor Agent</div>
            </div>
            <div class="discussion-content">
                "Excellent points from both perspectives. I suggest a phased approach: start with a focused MVP, validate user adoption, then scale. This reduces risk while maintaining growth potential."
            </div>
        </div>
    `;
}

// Generate Final Recommendation (fallback)
function generateFinalRecommendationContent() {
    const startup = currentStartupData;
    
    // Calculate overall score
    const overallScore = calculateOverallScore();
    const riskLevel = calculateRiskLevel();
    const marketPotential = calculateMarketPotential();
    const feasibility = calculateFeasibility();
    
    let recommendation = '';
    let nextSteps = [];
    
    if (overallScore >= 8) {
        recommendation = 'Strong recommendation to proceed with development and seek funding.';
        nextSteps = [
            'Develop detailed business plan',
            'Create MVP prototype',
            'Begin fundraising process',
            'Build core team'
        ];
    } else if (overallScore >= 6) {
        recommendation = 'Proceed with modifications and additional validation.';
        nextSteps = [
            'Refine business model',
            'Conduct additional market research',
            'Develop MVP with user feedback',
            'Consider pivot if necessary'
        ];
    } else {
        recommendation = 'Significant changes needed before proceeding.';
        nextSteps = [
            'Reassess problem-solution fit',
            'Conduct comprehensive market research',
            'Consider alternative approaches',
            'Validate core assumptions'
        ];
    }
    
    return `
        <h4>Final Assessment</h4>
        <p><strong>Overall Recommendation:</strong> ${recommendation}</p>
        
        <h4>Key Metrics</h4>
        <ul>
            <li><strong>Overall Score:</strong> ${overallScore}/10</li>
            <li><strong>Risk Level:</strong> ${riskLevel}</li>
            <li><strong>Market Potential:</strong> ${marketPotential}</li>
            <li><strong>Technical Feasibility:</strong> ${feasibility}</li>
        </ul>
        
        <h4>Critical Success Factors</h4>
        <ul>
            <li>${startup.solution.length > 100 ? 'Strong solution design' : 'Solution needs refinement'}</li>
            <li>${startup.businessModel.length > 50 ? 'Clear business model' : 'Business model requires development'}</li>
            <li>${startup.competitiveAdvantage.length > 50 ? 'Competitive differentiation' : 'Need stronger competitive advantage'}</li>
        </ul>
        
        <h4>Recommended Next Steps</h4>
        <ul>
            ${nextSteps.map(step => `<li>${step}</li>`).join('')}
        </ul>
        
        <h4>Timeline Recommendation</h4>
        <p>${overallScore >= 8 ? '6-12 months to MVP' : overallScore >= 6 ? '9-18 months to MVP' : '12-24 months to MVP'}</p>
    `;
}

// Display summary metrics (fallback)
function displaySummaryMetrics() {
    const overallScore = calculateOverallScore();
    const riskLevel = calculateRiskLevel();
    const marketPotential = calculateMarketPotential();
    const feasibility = calculateFeasibility();
    
    // Update score elements
    document.getElementById('overallScore').textContent = `${overallScore}/10`;
    document.getElementById('overallScore').className = `score ${getScoreClass(overallScore)}`;
    
    document.getElementById('riskLevel').textContent = riskLevel;
    document.getElementById('riskLevel').className = `risk-level ${getScoreClass(riskLevel === 'Low' ? 9 : riskLevel === 'Medium' ? 6 : 3)}`;
    
    document.getElementById('marketPotential').textContent = marketPotential;
    document.getElementById('marketPotential').className = `market-potential ${getScoreClass(marketPotential === 'High' ? 9 : marketPotential === 'Medium' ? 6 : 3)}`;
    
    document.getElementById('feasibility').textContent = feasibility;
    document.getElementById('feasibility').className = `feasibility ${getScoreClass(feasibility === 'High' ? 9 : feasibility === 'Medium' ? 6 : 3)}`;
}

// Helper functions for analysis (fallback)
function analyzeBusinessModel(businessModel) {
    const length = businessModel.length;
    if (length > 100) return 9;
    if (length > 70) return 7;
    if (length > 40) return 5;
    return 3;
}

function estimateMarketSize(targetMarket) {
    if (targetMarket.includes('global') || targetMarket.includes('worldwide')) return 50000000000;
    if (targetMarket.includes('national') || targetMarket.includes('country')) return 5000000000;
    if (targetMarket.includes('regional')) return 1000000000;
    return 500000000;
}

function analyzeCompetitiveAdvantage(advantage) {
    const length = advantage.length;
    if (length > 80) return 9;
    if (length > 50) return 7;
    if (length > 30) return 5;
    return 3;
}

function analyzeTechnicalFeasibility(solution) {
    if (solution.includes('AI') || solution.includes('machine learning')) return 'High - Leverages proven AI technologies';
    if (solution.includes('mobile') || solution.includes('web')) return 'High - Standard development practices';
    if (solution.includes('blockchain') || solution.includes('IoT')) return 'Medium - Emerging technologies';
    return 'Medium - Standard technical approach';
}

function analyzeMarketResearch(targetMarket) {
    return {
        validation: 'Moderate - Needs additional validation',
        trends: 'Growing market with increasing demand',
        growth: 'Expected 15-20% annual growth',
        challenges: 'Regulatory and competitive barriers exist'
    };
}

function analyzeCompetitiveLandscape(advantage) {
    return {
        intensity: 'High - Crowded market with established players'
    };
}

function analyzeUserExperience(solution) {
    return {
        quality: 'Good - Clear user value proposition',
        considerations: [
            'Focus on intuitive design',
            'Ensure mobile responsiveness',
            'Implement user feedback loops'
        ]
    };
}

function analyzeAdoptionPotential(targetMarket) {
    return {
        level: 'Medium - Requires strong marketing and UX',
        factors: [
            'Market education needed',
            'Competitive alternatives exist',
            'Clear value proposition'
        ]
    };
}

function analyzeUserValue(solution, problem) {
    return {
        strength: 'Strong - Directly addresses user pain points',
        benefits: [
            'Solves real user problems',
            'Improves efficiency',
            'Cost-effective solution'
        ]
    };
}

// Calculate overall metrics (fallback)
function calculateOverallScore() {
    const businessModel = analyzeBusinessModel(currentStartupData.businessModel);
    const competitiveAdvantage = analyzeCompetitiveAdvantage(currentStartupData.competitiveAdvantage);
    const marketSize = estimateMarketSize(currentStartupData.targetMarket);
    
    let score = (businessModel + competitiveAdvantage) / 2;
    
    if (marketSize > 10000000000) score += 1;
    if (currentStartupData.solution.length > 100) score += 0.5;
    if (currentStartupData.problemStatement.length > 100) score += 0.5;
    
    return Math.min(10, Math.round(score * 10) / 10);
}

function calculateRiskLevel() {
    const score = calculateOverallScore();
    if (score >= 8) return 'Low';
    if (score >= 6) return 'Medium';
    return 'High';
}

function calculateMarketPotential() {
    const marketSize = estimateMarketSize(currentStartupData.targetMarket);
    if (marketSize > 10000000000) return 'High';
    if (marketSize > 1000000000) return 'Medium';
    return 'Low';
}

function calculateFeasibility() {
    const technicalFeasibility = analyzeTechnicalFeasibility(currentStartupData.solution);
    if (technicalFeasibility.includes('High')) return 'High';
    if (technicalFeasibility.includes('Medium')) return 'Medium';
    return 'Low';
}

function getScoreClass(score) {
    if (score >= 8) return 'high';
    if (score >= 6) return 'medium';
    return 'low';
}

// Utility functions
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function scrollToSection(section) {
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function addSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

function showError(message) {
    // Create and show error notification
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        <span>${message}</span>
    `;
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ef4444;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    `;
    
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Action button functions
function downloadReport() {
    if (!currentStartupData) return;
    
    const report = generateReport();
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `startup-evaluation-${currentStartupData.startupName.replace(/\s+/g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function generateReport() {
    const startup = currentStartupData;
    const overallScore = calculateOverallScore();
    
    function stripHtml(html) {
        return typeof html === 'string' ? html.replace(/<[^>]*>/g, '') : '';
    }

    function serializeAgentAnalysis(agent) {
        // Simulation mode stored as HTML string
        if (typeof agent === 'string') {
            return stripHtml(agent);
        }
        // Real API object
        if (agent && typeof agent === 'object') {
            const lines = [];
            lines.push(`Agent: ${agent.agent || ''} (${agent.role || ''})`);
            if (typeof agent.relevant_documents === 'number') {
                lines.push(`Relevant documents: ${agent.relevant_documents}`);
            }
            if (Array.isArray(agent.document_previews) && agent.document_previews.length) {
                lines.push('Top documents:');
                for (const d of agent.document_previews.slice(0, 5)) {
                    const name = (d.source || '').toString().split('\\').pop();
                    const rel = d.relevance != null ? ` (relevance: ${Math.round(d.relevance * 100)}%)` : '';
                    lines.push(` - ${name}${rel}`);
                }
            }
            if (agent.llm_analysis) {
                lines.push('LLM Analysis:');
                lines.push(agent.llm_analysis);
            }
            if (Array.isArray(agent.insights) && agent.insights.length) {
                lines.push('Insights:');
                for (const i of agent.insights) lines.push(` - ${i}`);
            }
            if (Array.isArray(agent.recommendations) && agent.recommendations.length) {
                lines.push('Recommendations:');
                for (const r of agent.recommendations) lines.push(` - ${r}`);
            }
            if (agent.quantitative_model) {
                const qm = agent.quantitative_model;
                lines.push('Quantitative Model:');
                lines.push(` - Type: ${qm.type || ''}`);
                if (qm.success_probability != null) lines.push(` - Success Probability: ${qm.success_probability}%`);
            }
            return lines.join('\n');
        }
        return 'Analysis not available';
    }
    
    return `
STARTUP AI EVALUATION REPORT
============================

Startup: ${startup.startupName}
Date: ${new Date().toLocaleDateString()}
Overall Score: ${overallScore}/10

PROBLEM STATEMENT
${startup.problemStatement}

SOLUTION
${startup.solution}

TARGET MARKET
${startup.targetMarket}

BUSINESS MODEL
${startup.businessModel}

COMPETITIVE ADVANTAGE
${startup.competitiveAdvantage}

AI AGENT ANALYSES
=================

INVESTOR AGENT
${serializeAgentAnalysis(agentAnalyses.investor)}

RESEARCHER AGENT
${serializeAgentAnalysis(agentAnalyses.researcher)}

USER AGENT
${serializeAgentAnalysis(agentAnalyses.user)}

FINAL RECOMMENDATION
====================
${document.getElementById('recommendationSummary') ? document.getElementById('recommendationSummary').innerText : 'Recommendation not available'}

SUMMARY METRICS
===============
Overall Score: ${overallScore}/10
Risk Level: ${calculateRiskLevel()}
Market Potential: ${calculateMarketPotential()}
Feasibility: ${calculateFeasibility()}

Generated by StartupAI Multi-Agent Evaluation Platform
    `;
}

function resetForm() {
    // Reset form
    startupForm.reset();
    
    // Hide all sections
    agentsSection.style.display = 'none';
    collaborationSection.style.display = 'none';
    recommendationSection.style.display = 'none';
    resultsSection.style.display = 'none';
    
    // Reset agent analyses
    agentAnalyses = {
        investor: null,
        researcher: null,
        user: null
    };
    
    // Reset current startup data
    currentStartupData = null;
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
    
    // Reset all loading and analysis elements
    document.querySelectorAll('.loading-spinner').forEach(el => el.style.display = 'block');
    document.querySelectorAll('.analysis-content').forEach(el => {
        el.style.display = 'none';
        el.innerHTML = '';
    });
    document.querySelectorAll('.discussion-thread, .recommendation-summary').forEach(el => {
        el.style.display = 'none';
        el.innerHTML = '';
    });
}
