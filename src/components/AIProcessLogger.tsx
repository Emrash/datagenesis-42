
// import React, { useEffect, useState } from 'react';
// import { motion, AnimatePresence } from 'framer-motion';
// import { useWebSocket } from '../hooks/useWebSocket';
// import { Brain, Cpu, AlertCircle } from 'lucide-react';

// interface AIStep {
//   id: string;
//   title: string;
//   description: string;
//   icon: React.ComponentType<any>;
//   status: 'pending' | 'active' | 'completed' | 'error';
//   progress?: number;
//   timestamp: number;
// }

// interface AIProcessLoggerProps {
//   isVisible: boolean;
//   isGenerating?: boolean;
//   onClose?: () => void;
// }

// const AIProcessLogger: React.FC<AIProcessLoggerProps> = ({ isVisible, onClose }) => {
//   const [steps, setSteps] = useState<AIStep[]>([]);
  
//   const { isConnected, sendMessage } = useWebSocket('/ws/generation', {
//     onMessage: (message) => {
//       if (message.type === 'step_update') {
//         handleStepUpdate(message.data);
//       }
//     },
//     onError: () => {
//       console.log('WebSocket connection failed - this is expected if backend is not running');
//     }
//   });

//   const handleStepUpdate = (data: any) => {
//     setSteps(prev => {
//       const existing = prev.find(step => step.id === data.id);
//       if (existing) {
//         return prev.map(step => 
//           step.id === data.id 
//             ? { ...step, ...data }
//             : step
//         );
//       }
//       return [...prev, data];
//     });
//   };

//   useEffect(() => {
//     if (isVisible && isConnected) {
//       // Request current process status
//       sendMessage({ type: 'get_status' });
//     }
//   }, [isVisible, isConnected, sendMessage]);

//   const getStatusColor = (status: string) => {
//     switch (status) {
//       case 'completed': return 'text-green-400';
//       case 'active': return 'text-blue-400';
//       case 'error': return 'text-red-400';
//       default: return 'text-gray-400';
//     }
//   };

//   const getStatusIcon = (status: string) => {
//     switch (status) {
//       case 'completed': return '✓';
//       case 'active': return '⟳';
//       case 'error': return '✗';
//       default: return '○';
//     }
//   };

//   if (!isVisible) return null;

//   return (
//     <motion.div
//       className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
//       initial={{ opacity: 0 }}
//       animate={{ opacity: 1 }}
//       exit={{ opacity: 0 }}
//     >
//       <motion.div
//         className="bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto"
//         initial={{ scale: 0.9, y: 20 }}
//         animate={{ scale: 1, y: 0 }}
//         exit={{ scale: 0.9, y: 20 }}
//       >
//         <div className="flex items-center justify-between mb-6">
//           <div className="flex items-center gap-3">
//             <Brain className="w-6 h-6 text-purple-400" />
//             <h2 className="text-xl font-semibold text-white">AI Process Monitor</h2>
//           </div>
//           {onClose && (
//             <button
//               onClick={onClose}
//               className="text-gray-400 hover:text-white transition-colors"
//             >
//               ✕
//             </button>
//           )}
//         </div>

//         {!isConnected && (
//           <div className="mb-4 p-3 bg-yellow-500/20 border border-yellow-500/50 rounded-lg flex items-center gap-2">
//             <AlertCircle className="w-4 h-4 text-yellow-400" />
//             <span className="text-yellow-200 text-sm">
//               Backend connection not available - monitoring disabled
//             </span>
//           </div>
//         )}

//         <div className="space-y-4">
//           <AnimatePresence>
//             {steps.map((step, index) => (
//               <motion.div
//                 key={step.id}
//                 className="flex items-start gap-4 p-4 bg-gray-700/50 rounded-lg"
//                 initial={{ opacity: 0, x: -20 }}
//                 animate={{ opacity: 1, x: 0 }}
//                 transition={{ delay: index * 0.1 }}
//               >
//                 <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
//                   step.status === 'completed' ? 'bg-green-500/20' :
//                   step.status === 'active' ? 'bg-blue-500/20' :
//                   step.status === 'error' ? 'bg-red-500/20' :
//                   'bg-gray-500/20'
//                 }`}>
//                   <span className={getStatusColor(step.status)}>
//                     {getStatusIcon(step.status)}
//                   </span>
//                 </div>
                
//                 <div className="flex-1 min-w-0">
//                   <div className="flex items-center gap-2 mb-1">
//                     <step.icon className="w-4 h-4 text-gray-400" />
//                     <h3 className="text-sm font-medium text-white">{step.title}</h3>
//                   </div>
//                   <p className="text-xs text-gray-400 mb-2">{step.description}</p>
                  
//                   {step.progress !== undefined && (
//                     <div className="w-full bg-gray-600 rounded-full h-1.5">
//                       <motion.div
//                         className="bg-blue-500 h-1.5 rounded-full"
//                         initial={{ width: 0 }}
//                         animate={{ width: `${step.progress}%` }}
//                         transition={{ duration: 0.5 }}
//                       />
//                     </div>
//                   )}
//                 </div>
                
//                 <div className="text-xs text-gray-500">
//                   {new Date(step.timestamp).toLocaleTimeString()}
//                 </div>
//               </motion.div>
//             ))}
//           </AnimatePresence>
//         </div>

//         {steps.length === 0 && (
//           <div className="text-center py-8 text-gray-400">
//             <Cpu className="w-12 h-12 mx-auto mb-4 opacity-50" />
//             <p>No active AI processes</p>
//             {!isConnected && (
//               <p className="text-xs mt-2">Backend connection required for real-time monitoring</p>
//             )}
//           </div>
//         )}
//       </motion.div>
//     </motion.div>
//   );
// };

// export default AIProcessLogger;
