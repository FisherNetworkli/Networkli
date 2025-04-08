export interface Message {
  id: string;
  senderId: string;
  receiverId: string;
  content: string;
  timestamp: string;
  read: boolean;
  attachments?: {
    type: 'image' | 'file';
    url: string;
    name: string;
  }[];
} 