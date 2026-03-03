import { useEffect, useRef } from 'react';
import { MessageSquareText, ThumbsDown, ThumbsUp, X } from 'lucide-react';

type FeedbackRating = 'up' | 'down';

type FeedbackModalProps = {
  open: boolean;
  rating: FeedbackRating | null;
  comment: string;
  isSubmitting: boolean;
  errorMessage: string | null;
  onClose: () => void;
  onCommentChange: (value: string) => void;
  onRatingChange: (rating: FeedbackRating) => void;
  onSubmit: () => void;
};

export function FeedbackModal({
  open,
  rating,
  comment,
  isSubmitting,
  errorMessage,
  onClose,
  onCommentChange,
  onRatingChange,
  onSubmit,
}: FeedbackModalProps) {
  const primaryActionRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }

    const focusTimeoutId = window.setTimeout(() => {
      primaryActionRef.current?.focus();
    }, 0);

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !isSubmitting) {
        onClose();
      }
    };

    window.addEventListener('keydown', handleEscape);

    return () => {
      window.clearTimeout(focusTimeoutId);
      window.removeEventListener('keydown', handleEscape);
    };
  }, [isSubmitting, onClose, open]);

  if (!open) {
    return null;
  }

  const canSubmit = Boolean(rating) && !isSubmitting;

  return (
    <div className="feedback-modal-backdrop" role="presentation">
      <div
        className="feedback-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="feedback-modal-title"
      >
        <button
          type="button"
          className="feedback-dismiss-btn"
          onClick={onClose}
          disabled={isSubmitting}
          aria-label="Cerrar comentario"
        >
          <X size={18} />
        </button>

        <div className="feedback-modal-copy">
          <span className="feedback-modal-eyebrow">Tu opinión</span>
          <h2 id="feedback-modal-title">¿Te sirvió este resultado?</h2>
          <p>
            Ayuda a mejorar el reconocimiento con una reacción rápida.
          </p>
        </div>

        <div className="feedback-rating-row" role="group" aria-label="Seleccionar valoración">
          <button
            ref={primaryActionRef}
            type="button"
            className={`feedback-rating-btn ${rating === 'up' ? 'active' : ''}`}
            aria-pressed={rating === 'up'}
            onClick={() => onRatingChange('up')}
            disabled={isSubmitting}
          >
            <ThumbsUp size={18} />
            <span>Me sirvió</span>
          </button>

          <button
            type="button"
            className={`feedback-rating-btn ${rating === 'down' ? 'active' : ''}`}
            aria-pressed={rating === 'down'}
            onClick={() => onRatingChange('down')}
            disabled={isSubmitting}
          >
            <ThumbsDown size={18} />
            <span>No me sirvió</span>
          </button>
        </div>

        {rating && (
          <label className="feedback-comment-field">
            <span>
              <MessageSquareText size={16} />
              <span>Comentario opcional</span>
            </span>
            <textarea
              value={comment}
              onChange={(event) => onCommentChange(event.target.value.slice(0, 500))}
              rows={4}
              maxLength={500}
              placeholder="Cuéntanos qué pasó con el resultado."
              disabled={isSubmitting}
            />
          </label>
        )}

        {errorMessage && (
          <p className="feedback-error-message">{errorMessage}</p>
        )}

        <div className="feedback-action-row">
          <button
            type="button"
            className="secondary-btn"
            onClick={onClose}
            disabled={isSubmitting}
          >
            Ahora no
          </button>
          <button
            type="button"
            className="primary-btn"
            onClick={onSubmit}
            disabled={!canSubmit}
          >
            {isSubmitting ? 'Enviando...' : 'Enviar'}
          </button>
        </div>
      </div>
    </div>
  );
}
