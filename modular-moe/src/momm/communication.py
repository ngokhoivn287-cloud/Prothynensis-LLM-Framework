"""Peer communication protocol for Prothynesis."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class MessageType(Enum):
    PROBLEM = "PROBLEM"
    PLAN = "PLAN"
    HYPOTHESIS = "HYPOTHESIS"
    EVIDENCE = "EVIDENCE"
    CRITIQUE = "CRITIQUE"
    OBJECTION = "OBJECTION"
    COUNTERARGUMENT = "COUNTERARGUMENT"
    VERIFICATION_REQUEST = "VERIFICATION_REQUEST"
    VERIFICATION_RESULT = "VERIFICATION_RESULT"
    REVISION = "REVISION"
    CONFIDENCE = "CONFIDENCE"
    CONSENSUS = "CONSENSUS"
    FINAL_CANDIDATE = "FINAL_CANDIDATE"
    MEMORY_SHARE = "MEMORY_SHARE"
    TASK_UPDATE = "TASK_UPDATE"
    RECRUITMENT_REQUEST = "RECRUITMENT_REQUEST"
    STATUS = "STATUS"


@dataclass
class PeerMessage:
    """Structured peer-to-peer message."""
    message_id: str
    message_type: str
    task_id: str
    sender: str
    target: Optional[str]
    content_summary: str
    evidence_refs: List[str] = field(default_factory=list)
    memory_refs: List[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "PeerMessage":
        return cls(**d)


class PeerCommunicationProtocol:
    """
    Peer-to-peer communication protocol.
    
    Manages structured message exchange between models.
    """
    
    def __init__(self, model_id: str, task_id: str):
        self.model_id = model_id
        self.task_id = task_id
        self._messages: List[PeerMessage] = []
        self._message_index: Dict[str, PeerMessage] = {}
    
    def create_message(
        self,
        message_type: str,
        content_summary: str,
        target: Optional[str] = None,
        evidence_refs: Optional[List[str]] = None,
        memory_refs: Optional[List[str]] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PeerMessage:
        """Create a new peer message."""
        message_id = str(uuid.uuid4())
        
        message = PeerMessage(
            message_id=message_id,
            message_type=message_type,
            task_id=self.task_id,
            sender=self.model_id,
            target=target,
            content_summary=content_summary,
            evidence_refs=evidence_refs or [],
            memory_refs=memory_refs or [],
            confidence=confidence,
            metadata=metadata or {},
        )
        
        self._messages.append(message)
        self._message_index[message_id] = message
        
        return message
    
    def send_critique(
        self,
        target: str,
        claim: str,
        evidence: str,
        confidence: float = 0.8,
    ) -> PeerMessage:
        """Send a critique message."""
        return self.create_message(
            message_type=MessageType.CRITIQUE.value,
            content_summary=f"Critique of {target}: {claim}",
            target=target,
            evidence_refs=[evidence],
            confidence=confidence,
            metadata={"claim": claim},
        )
    
    def send_evidence(
        self,
        target: str,
        evidence_summary: str,
        confidence: float = 0.9,
    ) -> PeerMessage:
        """Send evidence."""
        return self.create_message(
            message_type=MessageType.EVIDENCE.value,
            content_summary=evidence_summary,
            target=target,
            confidence=confidence,
        )
    
    def send_revision(
        self,
        target: str,
        revision_summary: str,
        memory_refs: Optional[List[str]] = None,
        confidence: float = 0.85,
    ) -> PeerMessage:
        """Send a revised candidate."""
        return self.create_message(
            message_type=MessageType.REVISION.value,
            content_summary=revision_summary,
            target=target,
            memory_refs=memory_refs or [],
            confidence=confidence,
        )
    
    def send_verification_request(
        self,
        target: str,
        claim: str,
        evidence: str,
    ) -> PeerMessage:
        """Request verification from a peer."""
        return self.create_message(
            message_type=MessageType.VERIFICATION_REQUEST.value,
            content_summary=f"Verify: {claim}",
            target=target,
            evidence_refs=[evidence],
            confidence=0.0,
        )
    
    def send_verification_result(
        self,
        target: str,
        claim: str,
        verified: bool,
        confidence: float = 0.9,
    ) -> PeerMessage:
        """Send verification result."""
        return self.create_message(
            message_type=MessageType.VERIFICATION_RESULT.value,
            content_summary=f"Verified: {claim} -> {verified}",
            target=target,
            confidence=confidence,
            metadata={"verified": verified},
        )
    
    def share_memory(
        self,
        target: str,
        memory_summary: str,
        memory_refs: List[str],
        confidence: float = 0.8,
    ) -> PeerMessage:
        """Share memory with a peer."""
        return self.create_message(
            message_type=MessageType.MEMORY_SHARE.value,
            content_summary=memory_summary,
            target=target,
            memory_refs=memory_refs,
            confidence=confidence,
        )
    
    def get_messages(
        self,
        message_type: Optional[str] = None,
        sender: Optional[str] = None,
        target: Optional[str] = None,
        limit: int = 50,
    ) -> List[PeerMessage]:
        """Get filtered messages."""
        results = self._messages
        
        if message_type:
            results = [m for m in results if m.message_type == message_type]
        if sender:
            results = [m for m in results if m.sender == sender]
        if target:
            results = [m for m in results if m.target == target]
        
        return results[-limit:]
    
    def get_message_by_id(self, message_id: str) -> Optional[PeerMessage]:
        """Get a specific message."""
        return self._message_index.get(message_id)
    
    def get_critiques(self, target: Optional[str] = None) -> List[PeerMessage]:
        """Get all critique messages."""
        return self.get_messages(message_type=MessageType.CRITIQUE.value, target=target)
    
    def get_verifications(self) -> List[PeerMessage]:
        """Get all verification messages."""
        return self.get_messages(message_type=MessageType.VERIFICATION_RESULT.value)
    
    def get_revisions(self) -> List[PeerMessage]:
        """Get all revision messages."""
        return self.get_messages(message_type=MessageType.REVISION.value)
    
    def get_peer_memory_shares(self) -> List[PeerMessage]:
        """Get all memory share messages."""
        return self.get_messages(message_type=MessageType.MEMORY_SHARE.value)
    
    def clear_messages(self) -> int:
        """Clear all messages."""
        count = len(self._messages)
        self._messages.clear()
        self._message_index.clear()
        return count
    
    def get_message_stats(self) -> Dict[str, Any]:
        """Get message statistics."""
        type_counts = {}
        for msg in self._messages:
            type_counts[msg.message_type] = type_counts.get(msg.message_type, 0) + 1
        
        return {
            "total_messages": len(self._messages),
            "by_type": type_counts,
            "unique_senders": len(set(m.sender for m in self._messages)),
            "unique_targets": len(set(m.target for m in self._messages if m.target)),
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize communication state."""
        return {
            "model_id": self.model_id,
            "task_id": self.task_id,
            "messages": [m.to_dict() for m in self._messages],
            "stats": self.get_message_stats(),
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "PeerCommunicationProtocol":
        """Deserialize communication state."""
        protocol = cls(model_id=data["model_id"], task_id=data["task_id"])
        
        for msg_data in data.get("messages", []):
            message = PeerMessage.from_dict(msg_data)
            protocol._messages.append(message)
            protocol._message_index[message.message_id] = message
        
        return protocol
