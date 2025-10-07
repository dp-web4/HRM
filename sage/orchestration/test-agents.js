/**
 * Test SAGE Agents
 */

const EagleVisionAgent = require('./agents/vision/eagle-vision-agent');
const TrustEngineAgent = require('./agents/trust/trust-engine-agent');
const SNARCMemoryAgent = require('./agents/memory/snarc-memory-agent');

async function testAgents() {
  console.log('🧪 Testing SAGE Agents...\n');
  
  // Test Vision Agent
  console.log('Testing Eagle Vision Agent:');
  const visionAgent = new EagleVisionAgent({ model_path: '/mock/path' });
  const visionResult = await visionAgent.processImage('test.jpg');
  console.log('  Result:', visionResult);
  
  // Test Trust Engine
  console.log('\nTesting Trust Engine:');
  const trustEngine = new TrustEngineAgent({ 
    initial_trust: 0.5,
    update_rate: 0.1 
  });
  const trustResult = await trustEngine.evaluateTrust('vision', {
    expected: 0.9,
    actual: 0.85
  });
  console.log('  Result:', trustResult);
  
  // Test Memory Agent
  console.log('\nTesting SNARC Memory:');
  const memoryAgent = new SNARCMemoryAgent({
    buffer_size: 100,
    salience_threshold: 0.3
  });
  const memoryResult = await memoryAgent.store({
    data: 'test memory',
    surprise: 0.5,
    relevance: 0.7,
    affect: 0.6
  });
  console.log('  Result:', memoryResult);
  
  console.log('\n✅ All tests completed');
}

testAgents().catch(console.error);
