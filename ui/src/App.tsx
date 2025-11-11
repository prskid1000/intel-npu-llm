import { ChatHistoryProvider } from './contexts/ChatHistoryContext';
import { ChatContainer } from './components/ChatContainer';

function App() {
  return (
    <ChatHistoryProvider>
      <ChatContainer />
    </ChatHistoryProvider>
  );
}

export default App;

