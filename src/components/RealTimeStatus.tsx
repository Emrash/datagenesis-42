import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Wifi, 
  WifiOff, 
  Activity, 
  Zap, 
  Brain, 
  Shield, 
  Users,
  CheckCircle,
  AlertCircle,
  Clock
} from 'lucide-react';
import { useWebSocket } from '../hooks/useWebSocket';
import { ApiService } from '../lib/api';

interface BackendStatus {
  healthy: boolean;
  geminiStatus: 'online' | 'offline' | 'unknown';
  agentsActive: boolean;
  lastCheck: Date | null;
}

interface AgentStatus {
  name: string;
  status: 'active' | 'processing' | 'error' | 'idle';
  performance: number;
  lastUpdate: string;
}

export const RealTimeStatus: React.FC = () => {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>({
    healthy: false,
    geminiStatus: 'unknown',
    agentsActive: false,
    lastCheck: null
  });
  
  const [agents, setAgents] = useState<AgentStatus[]>([
    { name: 'Privacy Agent', status: 'idle', performance: 95, lastUpdate: '' },
    { name: 'Quality Agent', status: 'idle', performance: 92, lastUpdate: '' },
    { name: 'Domain Expert', status: 'idle', performance: 98, lastUpdate: '' },
    { name: 'Bias Detector', status: 'idle', performance: 88, lastUpdate: '' }
  ]);

  const { isConnected, lastMessage } = useWebSocket("generation");

  // Check backend health periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await ApiService.healthCheck();
        setBackendStatus({
          healthy: health.healthy,
          geminiStatus: health.data?.services?.gemini?.status === 'ready' ? 'online' : 'offline',
          agentsActive: health.data?.services?.agents === 'active',
          lastCheck: new Date()
        });
      } catch (error) {
        setBackendStatus(prev => ({
          ...prev,
          healthy: false,
          geminiStatus: 'offline',
          lastCheck: new Date()
        }));
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000); // Check every 10 seconds
    return () => clearInterval(interval);
  }, []);

  // Listen for agent updates via WebSocket
  useEffect(() => {
    if (lastMessage?.type === 'agent_update') {
      const { agent_id, status } = lastMessage.data;
      setAgents(prev => prev.map(agent => 
        agent.name.toLowerCase().includes(agent_id.toLowerCase()) 
          ? { ...agent, status: status.status, performance: status.performance, lastUpdate: new Date().toLocaleTimeString() }
          : agent
      ));
    }
  }, [lastMessage]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
      case 'active':
        return 'text-green-400 bg-green-500/20 border-green-500/30';
      case 'processing':
        return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
      case 'offline':
      case 'error':
        return 'text-red-400 bg-red-500/20 border-red-500/30';
      default:
        return 'text-gray-400 bg-gray-500/20 border-gray-500/30';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
      case 'active':
        return <CheckCircle className="w-4 h-4" />;
      case 'processing':
        return <Activity className="w-4 h-4 animate-pulse" />;
      case 'offline':
      case 'error':
        return <AlertCircle className="w-4 h-4" />;
      default:
        return <Clock className="w-4 h-4" />;
    }
  };

  return (
    <div className="space-y-4">
      {/* Backend Connection Status */}
      <motion.div
        className={`p-4 rounded-lg border ${getStatusColor(backendStatus.healthy ? 'active' : 'offline')}`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {backendStatus.healthy ? (
              <Wifi className="w-5 h-5 text-green-400" />
            ) : (
              <WifiOff className="w-5 h-5 text-red-400" />
            )}
            <span className="font-medium">Backend Connection</span>
          </div>
          <div className="flex items-center gap-2">
            {isConnected && (
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            )}
            <span className="text-sm">
              {backendStatus.healthy ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4" />
            <span>Gemini 2.0 Flash:</span>
            <span className={backendStatus.geminiStatus === 'online' ? 'text-green-400' : 'text-red-400'}>
              {backendStatus.geminiStatus === 'online' ? 'Online' : 'Offline'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Users className="w-4 h-4" />
            <span>AI Agents:</span>
            <span className={backendStatus.agentsActive ? 'text-green-400' : 'text-gray-400'}>
              {backendStatus.agentsActive ? 'Active' : 'Idle'}
            </span>
          </div>
        </div>

        {backendStatus.lastCheck && (
          <div className="mt-2 text-xs text-gray-400">
            Last checked: {backendStatus.lastCheck.toLocaleTimeString()}
          </div>
        )}
      </motion.div>

      {/* AI Agents Status */}
      <motion.div
        className="p-4 bg-gray-800/50 rounded-lg border border-gray-700"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1 }}
      >
        <div className="flex items-center gap-2 mb-3">
          <Zap className="w-5 h-5 text-purple-400" />
          <span className="font-medium text-white">AI Agents Status</span>
        </div>
        
        <div className="space-y-2">
          <AnimatePresence>
            {agents.map((agent, index) => (
              <motion.div
                key={agent.name}
                className="flex items-center justify-between p-2 bg-gray-700/30 rounded border border-gray-600/30"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2, delay: index * 0.05 }}
              >
                <div className="flex items-center gap-2">
                  {agent.name === 'Privacy Agent' && <Shield className="w-4 h-4 text-blue-400" />}
                  {agent.name === 'Quality Agent' && <CheckCircle className="w-4 h-4 text-green-400" />}
                  {agent.name === 'Domain Expert' && <Brain className="w-4 h-4 text-purple-400" />}
                  {agent.name === 'Bias Detector' && <Activity className="w-4 h-4 text-orange-400" />}
                  <span className="text-sm text-gray-300">{agent.name}</span>
                </div>
                
                <div className="flex items-center gap-2">
                  <div className="text-xs text-gray-400">
                    {agent.performance}%
                  </div>
                  <div className={`px-2 py-1 rounded text-xs border ${getStatusColor(agent.status)}`}>
                    <div className="flex items-center gap-1">
                      {getStatusIcon(agent.status)}
                      <span className="capitalize">{agent.status}</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </motion.div>

      {/* WebSocket Connection Status */}
      <motion.div
        className={`p-3 rounded-lg border text-sm ${isConnected 
          ? 'bg-green-500/10 border-green-500/30 text-green-300' 
          : 'bg-yellow-500/10 border-yellow-500/30 text-yellow-300'
        }`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.2 }}
      >
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'}`}></div>
          <span>
            {isConnected ? 'Real-time updates connected' : 'Connecting to real-time updates...'}
          </span>
        </div>
      </motion.div>
    </div>
  );
};

export default RealTimeStatus;