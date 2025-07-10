import React from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  Target,
  AlertTriangle,
  CheckCircle
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

const Analytics: React.FC = () => {
  const metrics = [
    { label: 'Data Quality', value: 94.7, change: 2.3, trend: 'up' },
    { label: 'Privacy Score', value: 98.2, change: 1.8, trend: 'up' },
    { label: 'Bias Detection', value: 92.1, change: -0.5, trend: 'down' },
    { label: 'Model Performance', value: 96.8, change: 3.2, trend: 'up' },
  ];

  const performanceData = [
    { name: 'Week 1', quality: 88, privacy: 92, bias: 85, performance: 89 },
    { name: 'Week 2', quality: 90, privacy: 94, bias: 87, performance: 91 },
    { name: 'Week 3', quality: 92, privacy: 96, bias: 89, performance: 93 },
    { name: 'Week 4', quality: 95, privacy: 98, bias: 92, performance: 97 },
  ];

  const domainMetrics = [
    { domain: 'Healthcare', quality: 96, privacy: 99, bias: 94, performance: 98 },
    { domain: 'Finance', quality: 94, privacy: 97, bias: 91, performance: 95 },
    { domain: 'Retail', quality: 92, privacy: 95, bias: 89, performance: 93 },
    { domain: 'Manufacturing', quality: 90, privacy: 94, bias: 87, performance: 91 },
  ];

  const radarData = [
    { subject: 'Data Quality', A: 95, B: 88, fullMark: 100 },
    { subject: 'Privacy', A: 98, B: 92, fullMark: 100 },
    { subject: 'Bias Mitigation', A: 92, B: 85, fullMark: 100 },
    { subject: 'Performance', A: 97, B: 89, fullMark: 100 },
    { subject: 'Scalability', A: 94, B: 87, fullMark: 100 },
    { subject: 'Reliability', A: 96, B: 91, fullMark: 100 },
  ];

  const alerts = [
    { 
      type: 'warning', 
      message: 'Bias detection score decreased in Finance domain',
      time: '5 minutes ago',
      severity: 'Medium'
    },
    { 
      type: 'success', 
      message: 'New privacy benchmark achieved in Healthcare',
      time: '12 minutes ago',
      severity: 'Low'
    },
    { 
      type: 'info', 
      message: 'Model performance improved by 3.2%',
      time: '1 hour ago',
      severity: 'Low'
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div 
        className="flex items-center justify-between"
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Analytics</h1>
          <p className="text-gray-400">Deep insights into your synthetic data generation performance</p>
        </div>
        <div className="flex gap-2">
          <button className="px-4 py-2 bg-gray-700/50 text-white rounded-lg hover:bg-gray-600/50 transition-colors">
            Export Report
          </button>
          <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 transition-all">
            Real-time View
          </button>
        </div>
      </motion.div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric, index) => (
          <motion.div
            key={index}
            className="p-6 bg-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl"
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                metric.trend === 'up' ? 'bg-gradient-to-br from-green-500 to-emerald-500' : 'bg-gradient-to-br from-red-500 to-orange-500'
              }`}>
                {metric.trend === 'up' ? <TrendingUp className="w-6 h-6 text-white" /> : <TrendingDown className="w-6 h-6 text-white" />}
              </div>
              <span className={`text-sm font-medium ${
                metric.trend === 'up' ? 'text-green-400' : 'text-red-400'
              }`}>
                {metric.trend === 'up' ? '+' : ''}{metric.change}%
              </span>
            </div>
            <h3 className="text-2xl font-bold text-white mb-1">{metric.value}%</h3>
            <p className="text-gray-400 text-sm">{metric.label}</p>
          </motion.div>
        ))}
      </div>

      {/* Performance Trends */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          className="p-6 bg-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl"
          initial={{ x: -50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <h3 className="text-xl font-semibold text-white mb-4">Performance Trends</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Line type="monotone" dataKey="quality" stroke="#8B5CF6" strokeWidth={2} />
              <Line type="monotone" dataKey="privacy" stroke="#10B981" strokeWidth={2} />
              <Line type="monotone" dataKey="bias" stroke="#F59E0B" strokeWidth={2} />
              <Line type="monotone" dataKey="performance" stroke="#3B82F6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.div
          className="p-6 bg-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl"
          initial={{ x: 50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <h3 className="text-xl font-semibold text-white mb-4">Multi-Agent Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <PolarRadiusAxis angle={90} domain={[0, 100]} />
              <Radar name="Current" dataKey="A" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.3} />
              <Radar name="Previous" dataKey="B" stroke="#10B981" fill="#10B981" fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Domain Analysis */}
      <motion.div
        className="p-6 bg-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl"
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.6 }}
      >
        <h3 className="text-xl font-semibold text-white mb-4">Domain Performance Analysis</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={domainMetrics}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="domain" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1F2937', 
                border: '1px solid #374151',
                borderRadius: '8px'
              }}
            />
            <Bar dataKey="quality" fill="#8B5CF6" />
            <Bar dataKey="privacy" fill="#10B981" />
            <Bar dataKey="bias" fill="#F59E0B" />
            <Bar dataKey="performance" fill="#3B82F6" />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Alerts & Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          className="p-6 bg-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl"
          initial={{ x: -50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.7 }}
        >
          <h3 className="text-xl font-semibold text-white mb-4">Alerts & Notifications</h3>
          <div className="space-y-3">
            {alerts.map((alert, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-gray-700/30 rounded-lg">
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  alert.type === 'warning' ? 'bg-yellow-400' :
                  alert.type === 'success' ? 'bg-green-400' : 'bg-blue-400'
                }`}></div>
                <div className="flex-1">
                  <p className="text-white text-sm">{alert.message}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-gray-400 text-xs">{alert.time}</span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      alert.severity === 'High' ? 'bg-red-500/20 text-red-300' :
                      alert.severity === 'Medium' ? 'bg-yellow-500/20 text-yellow-300' :
                      'bg-green-500/20 text-green-300'
                    }`}>
                      {alert.severity}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div
          className="p-6 bg-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl"
          initial={{ x: 50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.8 }}
        >
          <h3 className="text-xl font-semibold text-white mb-4">AI Insights</h3>
          <div className="space-y-4">
            <div className="p-4 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-lg border border-purple-500/20">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-5 h-5 text-purple-400" />
                <span className="text-purple-300 font-medium">Optimization Opportunity</span>
              </div>
              <p className="text-gray-300 text-sm">
                Healthcare domain shows 99% privacy compliance. Consider applying similar techniques to Finance domain.
              </p>
            </div>
            
            <div className="p-4 bg-gradient-to-r from-green-500/10 to-emerald-500/10 rounded-lg border border-green-500/20">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span className="text-green-300 font-medium">Performance Achievement</span>
              </div>
              <p className="text-gray-300 text-sm">
                Multi-agent system achieved 97% efficiency in cross-domain knowledge transfer.
              </p>
            </div>
            
            <div className="p-4 bg-gradient-to-r from-orange-500/10 to-red-500/10 rounded-lg border border-orange-500/20">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-5 h-5 text-orange-400" />
                <span className="text-orange-300 font-medium">Attention Required</span>
              </div>
              <p className="text-gray-300 text-sm">
                Bias detection agent suggests reviewing Manufacturing domain data distribution.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Analytics;